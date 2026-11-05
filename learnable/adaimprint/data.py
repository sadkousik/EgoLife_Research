"""Trace I/O: load recorded AdaImprint traces, split them by day, attach QA evidence.

A trace (written by ``scripts/record_trace.py``) holds, per FIO in stream order:
static features, contextual embedding, clip timestamp, prototype id, the
heuristic importance w(f_t) and the best same-pattern memory candidate with
its similarity.
"""

import json
import os
import re
from typing import Dict, Optional

import torch

from .model import FEATURE_NAMES

RERECORD_HINT = "re-record it with `python -m learnable.scripts.record_trace`"


# ----------------------------------------------------------------------
# Phase-1 cache
# ----------------------------------------------------------------------
def iter_caption_fios(captions_path: str, cache_dir: str, target_day: int):
    with open(captions_path) as f:
        captions = json.load(f)
    valid = {f"DAY{i}" for i in range(1, target_day + 1)}
    captions = [c for c in captions if c.get("date") in valid]   # same order as scripts/build_memory.py
    missing = 0
    for c in captions:
        path = os.path.join(cache_dir, f"{c['date']}-{c['start_time']}.json")
        fios = []
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    fios = json.load(fh) or []
            except (json.JSONDecodeError, OSError):
                fios = []
        else:
            missing += 1
        yield c, fios
    if missing:
        print(f"[trace] {missing}/{len(captions)} captions had no cache file")


# ----------------------------------------------------------------------
# Traces
# ----------------------------------------------------------------------
def compute_candidates(emb, pattern, window: int, sim_thr: float):
    """Best earlier FIO with the same action|object|app and ctx sim >= thr, within window."""
    T = emb.shape[0]
    cand = torch.full((T,), -1, dtype=torch.long)
    sim = torch.zeros(T)
    for t in range(1, T):
        lo = max(0, t - window)
        same = torch.nonzero(pattern[lo:t] == pattern[t]).squeeze(1)
        if same.numel():
            sims = emb[lo + same] @ emb[t]
            j = int(sims.argmax())
            if sims[j] >= sim_thr:
                cand[t] = lo + int(same[j])
                sim[t] = sims[j]
    return cand, sim


def heuristic_valid(cand, cand_sim, keep, thr):
    """The heuristic's consolidation rule: kept candidate with sim >= thr."""
    return (cand >= 0) & (cand_sim >= thr) & keep[cand.clamp_min(0)]


def load_trace(path: str, window: int = 100, sim_thr: float = 0.75) -> Dict:
    b = torch.load(path, map_location="cpu", weights_only=False)
    if list(b["feature_names"]) != list(FEATURE_NAMES) or "pattern" not in b:
        raise ValueError(f"{path}: recorded with another code version; {RERECORD_HINT}")
    emb = b["emb"].float()
    emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
    ts = b["clip_ts"]
    meta = b.get("meta", {})
    ok = meta.get("opt_kwargs", {})
    st = b["stats"]
    cons_thr = ok.get("compression_sim_threshold", 0.75)
    if "cand" in b and "cand_sim" in b:
        cand, cand_sim, floor = b["cand"], b["cand_sim"].float(), float(b["cand_floor"])
    elif "cand" in b:
        # older trace: only candidates with sim >= the heuristic threshold were recorded
        cand = b["cand"]
        cand_sim = torch.where(cand >= 0, (emb * emb[cand.clamp_min(0)]).sum(1), torch.zeros(()))
        floor = cons_thr
        print(f"[warn] {path}: no candidate similarities below {cons_thr}; the consolidation "
              f"threshold can only rise; {RERECORD_HINT} for the full range.")
    else:
        floor = 0.5
        cand, cand_sim = compute_candidates(emb, b["pattern"], window, floor)
    return {
        "path": path, "name": meta.get("wearer") or os.path.basename(path),
        "feats": b["feats"].float(), "emb": emb,
        "ts": (ts - ts[0]).float(), "abs_ts": ts,
        "proto": b["proto"].long(), "base_w": b["base_w"].float(),
        "centroids": b["centroids"].float(),
        "cand": cand, "cand_sim": cand_sim, "cand_floor": floor, "cons_thr": cons_thr,
        "tau": ok.get("tau_threshold"), "lambda": ok.get("lambda_decay"),
        "max_prototypes": ok.get("max_prototypes", 300),
        "heur_write_rate": st["total_kept"] / max(1, st["total_input"]),
        "evidence": None,
    }


def split_by_day(tr: Dict, day: int):
    """Split one wearer's trace into days < day (train) and days >= day (validation)."""
    days = (tr["abs_ts"] // 86400).long() + 1
    idx = int((days < day).sum())
    if not (0 < idx < len(days)) or not bool((days[:idx] < day).all()):
        raise ValueError(f"{tr['name']}: cannot split at DAY{day} "
                         f"(trace covers DAY{int(days.min())}-DAY{int(days.max())})")

    def part(lo, hi, tag):
        p = {k: tr[k] for k in ("centroids", "tau", "lambda", "max_prototypes",
                                "cand_floor", "cons_thr")}
        for k in ("feats", "emb", "abs_ts", "proto", "base_w", "cand_sim"):
            p[k] = tr[k][lo:hi]
        p["ts"] = (p["abs_ts"] - p["abs_ts"][0]).float()
        c = tr["cand"][lo:hi] - lo
        p["cand"] = torch.where(c >= 0, c, torch.full_like(c, -1))
        keep = p["base_w"] >= (p["tau"] if p["tau"] is not None else 0.2)
        valid = heuristic_valid(p["cand"], p["cand_sim"], keep, p["cons_thr"])
        p["heur_write_rate"] = float((keep & ~valid).float().mean())
        p["evidence"] = None
        p["path"] = f"{tr['path']}#{tag}"
        p["name"] = f"{tr['name']}[{tag}]"
        return p

    return part(0, idx, f"DAY<{day}"), part(idx, len(days), f"DAY>={day}")


def _qa_time_to_sec(t) -> Optional[float]:
    if not isinstance(t, dict) or "date" not in t or "time" not in t:
        return None
    m = re.search(r"(\d+)", str(t["date"]))
    s = str(t["time"]).zfill(8)
    if not m or len(s) < 6:
        return None
    return (int(m.group(1)) - 1) * 86400 + int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])


def attach_evidence(trace: Dict, qa_path: str, window: float) -> None:
    with open(qa_path) as f:
        items = json.load(f)
    targets = []
    for it in items:
        tt = it.get("target_time")
        for t in (tt if isinstance(tt, list) else [tt]):
            s = _qa_time_to_sec(t)
            if s is not None:
                targets.append(s)
    if not targets:
        raise ValueError(f"{qa_path}: no usable target_time fields ({{date, time}}) found")
    tg = torch.tensor(sorted(targets), dtype=torch.float64)
    d = (trace["abs_ts"][:, None] - tg[None, :]).abs().min(dim=1).values
    trace["evidence"] = d <= window
    print(f"[evidence] {trace['name']}: {len(targets)} targets -> "
          f"{int(trace['evidence'].sum())} positive FIOs (±{window:.0f}s)")


def padded_centroids(tr, n_protos: int) -> torch.Tensor:
    C = torch.zeros(n_protos, tr["centroids"].shape[1])
    C[: tr["centroids"].shape[0]] = tr["centroids"]
    return C
