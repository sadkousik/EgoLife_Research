"""AdaImprint inside the production FIO build.

``LearnableFIOOptimizer`` subclasses ``fio.optimizer.FIOOptimizer`` (FIO/fio/optimizer.py)
without changing it. The parent still does steps 1, 2 and 7 of the algorithm
(encode, prototype matching, prototype statistics for *every* FIO, history
buffer); prototype statistics stay separate from the episodic memory M
(``kept_output``). The subclass adds:

* step 3   pre-update features (``_features``), recorded for training traces;
* step 4-5 surprise, controller, adaptive importance and write/merge/update
           preferences (via ``memory.StreamProcessor``);
* step 6   the decision: discard (I_adapt < tau), else write a new entry, merge
           into the equivalent entry, or update it; under a hard budget B a write
           into a full memory evicts the lowest-utility entry
           (utility = I_adapt * exp(-lambda * age)), consolidating it into a
           live entry with the same action|object|app when one exists so its
           temporal evidence survives, or discards the incoming FIO when it is
           the least useful;
* step 8   frozen by default; ``online_lr > 0`` keeps training the controller
           and predictor on the stream (chunked truncated BPTT).

With ``model=None`` it behaves exactly like FIOOptimizer and only records a
trace (used by ``learnable/scripts/record_trace.py``).
"""

import contextlib
import copy
import io
import json
import math
import os
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_FIO_DIR = Path(os.environ.get(
    "FIO_SRC_DIR", Path(__file__).resolve().parents[2] / "FIO"))
if str(_FIO_DIR) not in sys.path:
    sys.path.insert(0, str(_FIO_DIR))

from fio.optimizer import (  # noqa: E402
    FIOOptimizer, _field_richness_score, _is_empty, _merge_fios, fio_to_seconds,
)

from .memory import StreamProcessor  # noqa: E402
from .model import FEATURE_NAMES, LearnableImportance, load_checkpoint  # noqa: E402

# Locked full-model config, copied from FIO/scripts/build_memory.py BASE_OPT_KWARGS so traces can
# be recorded without importing fio_v3 (which loads Qwen at import time).
# Same locked config as FIO/ablation/configs.py, without growth tracking
# (the trace replays streams many times; growth logs are not needed).
from ablation.configs import BASE_OPT_KWARGS  # noqa: E402
FULL_OPT_KWARGS = {**BASE_OPT_KWARGS, "track_growth": False}

# Fields an "update" never takes from the incoming FIO.
_UPDATE_KEEP = {"id", "date", "start_time", "end_time", "video_path", "proto_id",
                "importance_score", "heuristic_score", "recency_score",
                "alpha_score",   # legacy: old Phase-1 caches still carry it
                "optimized", "action_sequence", "embedding", "document"}


def _nonempty(v) -> bool:
    return v not in (None, "", [], {}, "null", "{}", "[]", "none", "None")


def _n_persons(v) -> int:
    if isinstance(v, list):
        return len([p for p in v if _nonempty(p)])
    if isinstance(v, str) and _nonempty(v):
        return len([p for p in v.replace(";", ",").split(",") if p.strip()])
    return 0


def _seq(entry: Dict) -> List:
    s = entry.get("action_sequence")
    if isinstance(s, str):
        try:
            s = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            s = None
    return list(s) if s else []


class LearnableFIOOptimizer(FIOOptimizer):
    def __init__(self, *args, model: Optional[LearnableImportance] = None,
                 memory_budget: Optional[int] = None, online_lr: float = 0.0,
                 online_chunk: int = 128, record_trace: bool = True, quiet: bool = False,
                 cand_floor: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Candidates (same-pattern memory entries) are searched down to this
        # similarity; the learned consolidation gate decides which ones count.
        if cand_floor is None:
            cand_floor = model.cfg.cons_floor if model is not None else 0.5
        self.cand_floor = min(float(cand_floor), self.COMPRESSION_SIM)
        self.lmodel = model
        self.memory_budget = memory_budget
        self.record_trace = record_trace
        self.quiet = quiet
        self._history_window = kwargs.get("history_window", 5)
        self._t = 0
        self._last_assign = (0.0, True)
        self._cur_action_emb = None
        self._decision = None
        self._pending: List[tuple] = []
        self._live = 0
        self.evictions = {"consolidated": 0, "dropped": 0, "incoming_discarded": 0}
        self.trace: Dict[str, List[Any]] = {k: [] for k in (
            "feats", "emb", "clip_ts", "proto", "pattern", "cand", "cand_sim", "ids",
            "learned_w", "base_w")}

        self.proc = None
        self.online_opt = None
        self.online_chunk = online_chunk
        if model is not None:
            cfg = model.cfg
            if cfg.n_protos < self._max_prototypes:
                raise ValueError(f"model predicts {cfg.n_protos} prototypes but the "
                                 f"optimizer allows {self._max_prototypes}")
            if abs(cfg.tau - self.KEEP_THRESHOLD) > 1e-9:
                raise ValueError(f"model tau={cfg.tau} != tau_threshold={self.KEEP_THRESHOLD}")
            budget = memory_budget or cfg.write_rate * cfg.expected_stream_len
            online = online_lr > 0
            if online:
                model.train()
                self.online_opt = torch.optim.Adam(model.parameters(), lr=online_lr)
            else:
                model.eval()
            self.proc = StreamProcessor(model, self._centroids, budget=budget, training=online)
            self._budget_for_occ = float(budget)

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs):
        model, _ = load_checkpoint(path)
        return cls(model=model, **kwargs)

    # -------- logging --------------------------------------------------
    def debug(self, title, data=None):
        if not self.quiet:
            super().debug(title, data)

    def process(self, fios: List[Dict]) -> Dict:
        n = max(1, len(fios))
        for i, f in enumerate(fios):
            f["_clip_pos"] = i / n
        if self.quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                res = super().process(fios)
        else:
            res = super().process(fios)
        self._flush()
        return res

    # -------- parent hooks ------------------------------------------------
    def normalize_action(self, fio: Dict):
        self._flush()
        emb = super().normalize_action(fio)
        self._cur_action_emb = emb
        return emb

    def assign_prototype(self, emb) -> int:
        if self.prototypes:
            P = np.stack(self.prototypes)
            sims = P @ emb / (np.linalg.norm(P, axis=1) * np.linalg.norm(emb) + 1e-8)
            best = float(sims.max())
        else:
            best = 0.0
        n_before = len(self.prototypes)
        pid = super().assign_prototype(emb)
        self._last_assign = (best, len(self.prototypes) > n_before)
        return pid

    def _find_consolidation_target(self, fio, proto_id, action_emb, ctx_emb):
        if self.lmodel is None or self._decision is None:
            return super()._find_consolidation_target(fio, proto_id, action_emb, ctx_emb)
        action, tgt, score = self._decision
        self._decision = None
        if action == "write" or tgt is None:
            return None
        ts = fio["_effective_ts"]
        if action == "update":
            snap = {k: copy.deepcopy(v) for k, v in fio.items()
                    if not k.startswith("_") and k not in _UPDATE_KEEP}
            self._pending.append(("update", tgt, snap, np.asarray(ctx_emb), score, ts))
        else:
            self._pending.append(("touch", tgt, ts))
        return tgt

    # -------- step 3: features ------------------------------------------
    def _features(self, proto_id: int, fio: Dict, base: Dict):
        bd = base["breakdown"]
        e_t = fio.get("_e_t")
        sims = []
        if e_t is not None:
            for h in self.history:
                e_j = h.get("_e_t") if isinstance(h, dict) else None
                if e_j is not None:
                    sims.append(self.cosine_sim(e_t, e_j))
        clip_ts = fio_to_seconds(fio)
        tod = (clip_ts % 86400) / 86400.0
        day = clip_ts // 86400 + 1
        dt = (max(0.0, fio["_effective_ts"] - self.proto_stats[proto_id]["last_seen"])
              if proto_id in self.proto_stats else 0.0)
        best_sim, is_new = self._last_assign
        row = {
            "log1p_freq": math.log(1 + bd["f (frequency)"]),
            "dt_sec": float(dt),
            "log1p_n_protos": math.log(1 + bd["df"]),
            "log_base_w": math.log(max(base["score"], 1e-12)),
            "novelty": 1.0 - best_sim,
            "is_new_proto": float(is_new),
            "hist_max_sim": max(sims) if sims else 0.0,
            "hist_related_frac": sum(1 for s in sims if s > self.DELTA_H) / max(1, self._history_window),
            "field_richness": _field_richness_score(fio),
            "has_speech": float(_nonempty(fio.get("speech_content"))),
            "n_persons": float(_n_persons(fio.get("persons"))),
            "has_tool_or_app": float(_nonempty(fio.get("tool")) or _nonempty(fio.get("app"))),
            "tod_sin": math.sin(2 * math.pi * tod),
            "tod_cos": math.cos(2 * math.pi * tod),
            "day_frac": (day - 1) / 7.0,
            "clip_pos": float(fio.get("_clip_pos", 0.0)),
        }
        return np.array([row[k] for k in FEATURE_NAMES], dtype=np.float32), clip_ts

    # -------- steps 4-6 ---------------------------------------------------
    def compute_weight(self, proto_id: int, fio: Dict) -> Dict[str, Any]:
        base = super().compute_weight(proto_id, fio)
        feats, clip_ts = self._features(proto_id, fio, base)
        ctx = np.asarray(fio["_ctx_emb"], dtype=np.float32)
        origin = self._t
        fio["_origin"] = origin
        fio["_last_ts"] = fio["_effective_ts"]
        score = base["score"]

        # best same-pattern entry in the current memory M, searched down to the
        # floor (the parent's rule with compression_sim lowered); the parent's
        # argmax is unchanged, so with sim >= 0.75 this is the heuristic's target
        tgt, cand_sim = self._find_candidate(fio, proto_id, ctx)
        cand_origin = int(self.kept_output[tgt].get("_origin", -1)) if tgt is not None else -1
        if self.lmodel is not None:
            score = self._learned_step(proto_id, fio, feats, ctx, clip_ts, tgt, cand_origin,
                                       cand_sim)
            base["breakdown"]["learned_w"] = round(score, 6)

        self._t += 1
        if self.record_trace:
            tr = self.trace
            tr["feats"].append(feats)
            tr["emb"].append(ctx.astype(np.float16))
            tr["clip_ts"].append(float(clip_ts) + 0.01 * float(fio.get("_clip_pos", 0.0)))
            tr["proto"].append(int(proto_id))
            tr["pattern"].append(zlib.crc32(self._struct_key(fio).encode()))
            tr["cand"].append(cand_origin)
            tr["cand_sim"].append(cand_sim)
            tr["ids"].append(str(fio.get("id", "")))
            tr["base_w"].append(float(base["score"]))
            tr["learned_w"].append(float(score))
        return {"score": score, "breakdown": base["breakdown"]}

    def _find_candidate(self, fio, proto_id, ctx):
        saved = self.COMPRESSION_SIM
        self.COMPRESSION_SIM = self.cand_floor
        try:
            tgt = super()._find_consolidation_target(fio, proto_id, self._cur_action_emb, ctx)
        finally:
            self.COMPRESSION_SIM = saved
        if tgt is None:
            return None, 0.0
        return tgt, float(self.cosine_sim(ctx, self.kept_output[tgt]["_ctx_emb"]))

    def _learned_step(self, proto_id, fio, feats, ctx, clip_ts, tgt, cand_origin,
                      cand_sim) -> float:
        live_before = self._live
        now = fio["_effective_ts"]

        def decide(keep, action, out):
            if (keep and action == 0 and self.memory_budget is not None
                    and self._live >= self.memory_budget):
                u_in = float(torch.exp(out["log_w"][0]))
                victim, u_min = self._lowest_utility(now)
                if victim is None or u_in <= u_min:
                    self.evictions["incoming_discarded"] += 1
                    return False, 0
                self._evict(victim)
            return keep, action

        ctxm = contextlib.nullcontext() if self.online_opt is not None else torch.no_grad()
        with ctxm:
            res = self.proc.step(torch.from_numpy(feats), torch.from_numpy(ctx), float(clip_ts),
                                 int(proto_id), cand_origin, cand_sim=cand_sim,
                                 occupancy=live_before / self._budget_for_occ,
                                 cand_valid=tgt is not None, decide=decide)
        score = math.exp(res["log_w"])
        if not res["keep"]:
            score = min(score, self.KEEP_THRESHOLD * 0.5)     # budget may veto a keep
        else:
            score = max(score, self.KEEP_THRESHOLD)
            if res["action"] == "write":
                self._live += 1
        self._decision = (res["action"], tgt, score)

        if self.online_opt is not None and self.proc.t % self.online_chunk == 0:
            losses = self.proc.chunk_loss()
            self.online_opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.lmodel.parameters(), 1.0)
            self.online_opt.step()
            self.proc.end_chunk()
        elif self.online_opt is None and self.proc.t % self.online_chunk == 0:
            self.proc.end_chunk()       # trims the processor's ring
        return score

    # -------- memory maintenance -------------------------------------------
    def _centroids(self) -> torch.Tensor:
        n = self.lmodel.cfg.n_protos
        C = torch.zeros(n, self.lmodel.cfg.emb_dim)
        if self.prototypes:
            P = torch.from_numpy(np.stack(self.prototypes).astype(np.float32))
            C[: P.shape[0]] = P / P.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return C

    def _flush(self) -> None:
        for op in self._pending:
            if op[0] == "touch":
                _, i, ts = op
                self.kept_output[i]["_last_ts"] = ts
            else:
                _, i, snap, ctx, score, ts = op
                entry = self.kept_output[i]
                for k, v in snap.items():
                    if not _is_empty(v):
                        entry[k] = v
                entry["_ctx_emb"] = ctx
                entry["importance_score"] = score
                entry["_last_ts"] = ts
        self._pending.clear()

    def _lowest_utility(self, now: float):
        lam = float(self.lmodel.lam)
        best, best_u = None, float("inf")
        for i, e in enumerate(self.kept_output):
            if e.get("_evicted"):
                continue
            u = float(e.get("importance_score", 0.0)) * math.exp(-lam * max(0.0, now - e.get("_last_ts", now)))
            if u < best_u:
                best, best_u = i, u
        return best, best_u

    def _evict(self, i: int) -> None:
        self._flush()
        victim = self.kept_output[i]
        key = self._struct_key(victim)
        nb = None
        for j in range(len(self.kept_output) - 1, -1, -1):
            e = self.kept_output[j]
            if j != i and not e.get("_evicted") and self._struct_key(e) == key:
                nb = j
                break
        if nb is not None:
            merged = _merge_fios(self.kept_output[nb], victim)
            if _seq(victim):                       # keep the victim's whole merge chain
                merged["action_sequence"] = merged["action_sequence"][:-1] + _seq(victim)
            merged["action_sequence"].sort(key=lambda s: str(s.get("start_time") or ""))
            self.kept_output[nb] = merged
            self._compression_log[nb]["collapsed"].append(
                {"raw_fio": {k: v for k, v in victim.items() if not k.startswith("_")},
                 "reason": "evicted-consolidated"})
            self._compression_log[nb]["collapsed"].extend(self._compression_log[i]["collapsed"])
            self.evictions["consolidated"] += 1
        else:
            self.proc.ring.scale_write(int(victim.get("_origin", -1)), 0.0)
            self.evictions["dropped"] += 1
        victim["_evicted"] = True
        victim.pop("_ctx_emb", None)          # no longer a consolidation target
        victim.pop("_action_emb", None)
        self._live -= 1

    def final_memory(self) -> List[Dict]:
        """Live episodic memory M after the stream (use this for storage)."""
        self._flush()
        out = []
        for i, e in enumerate(self.kept_output):
            if e.get("_evicted"):
                continue
            clean = {k: v for k, v in e.items() if not k.startswith("_") and k != "embedding"}
            if not self._compression_log[i]["collapsed"] and not clean.get("action_sequence"):
                clean.pop("action_sequence", None)
            out.append(clean)
        return out

    def memory_report(self) -> Dict:
        rep = {"live_entries": len(self.final_memory()), "evictions": dict(self.evictions)}
        if self.proc is not None:
            rep.update(self.proc.stats())
        return rep

    # -------- trace export -------------------------------------------------
    def export_trace(self, path: str, meta: Optional[Dict] = None) -> Dict:
        tr = self.trace
        if not tr["feats"]:
            raise RuntimeError("empty trace — no FIOs were processed")
        P = np.stack(self.prototypes).astype(np.float32)
        P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
        blob = {
            "feature_names": list(FEATURE_NAMES),
            "feats": torch.from_numpy(np.stack(tr["feats"])),
            "emb": torch.from_numpy(np.stack(tr["emb"])),
            "clip_ts": torch.tensor(tr["clip_ts"], dtype=torch.float64),
            "proto": torch.tensor(tr["proto"], dtype=torch.long),
            "pattern": torch.tensor(tr["pattern"], dtype=torch.long),
            # origin of the equivalent memory entry under the heuristic's memory (-1: none)
            "cand": torch.tensor(tr["cand"], dtype=torch.long),
            # its contextual similarity; candidates go down to cand_floor
            "cand_sim": torch.tensor(tr["cand_sim"], dtype=torch.float32),
            "cand_floor": self.cand_floor,
            "base_w": torch.tensor(tr["base_w"], dtype=torch.float64),
            "learned_w": torch.tensor(tr["learned_w"], dtype=torch.float64),
            "centroids": torch.from_numpy(P),
            "ids": tr["ids"],
            "stats": dict(self._stats),
            "meta": meta or {},
        }
        torch.save(blob, path)
        return blob
