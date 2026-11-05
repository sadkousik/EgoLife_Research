"""Standalone FIO memory build with AdaImprint (Phase 2 only).

A learnable-path copy of ``FIO/scripts/build_memory.py``'s Phase 2 that lives in
``learnable/scripts/`` so the production file stays untouched. It reads the existing
Phase-1 FIO cache, runs ``LearnableFIOOptimizer`` and writes the same Chroma
stores and ``build_report.json`` that ``FIO/scripts/qa.py`` reads. It never imports
``fio.llm`` (no Qwen, no GPU): Phase 1 / LLM extraction still runs through
``FIO/scripts/build_memory.py``.

    # learned build for one wearer
    python -m learnable.scripts.build_memory \
        --ckpt learnable/outputs/ckpt_full_valA5_KATRINA/best.pt \
        --captions generated_caption/A5_KATRINA_meta.json \
        --cache-dir FIO_mem/A5_KATRINA/fio_cache \
        --out FIO_mem/A5_KATRINA_learnable_full

    # same script without --ckpt = the heuristic FIOOptimizer, built exactly as FIO/scripts/build_memory.py
    python -m learnable.scripts.build_memory --captions ... --cache-dir ... --out FIO_mem/A5_KATRINA_heuristic

Used from the FIO/fio package (imported, not copied):
    fio.optimizer.FIOOptimizer        (through learnable.adaimprint.fio_adapter)
    fio.storage.dual.DualFIOStorage   -> fio.storage.chroma, fio.schema
    fio.schema.compute_recency_for_day
    fio.tracking.BuildTracker
    fio.storage.staging.StagedChromaDB
    fio.stats.analyze_chroma_dates
Copied from FIO/scripts/build_memory.py (it cannot be imported without loading Qwen):
    _action_sequence_to_text, build_document_string_opt_aware,
    the per-caption Phase-2 loop and the recency step.
"""

import argparse
import json
import os
import re
from typing import Dict, List

# fio_adapter puts FIO/ on sys.path, so it must be imported before any `fio.*` module
from learnable.adaimprint.fio_adapter import FULL_OPT_KWARGS, LearnableFIOOptimizer
from learnable.adaimprint.data import iter_caption_fios

from fio.schema import compute_recency_for_day         # noqa: E402
from fio.storage.dual import DualFIOStorage            # noqa: E402
from fio.storage.staging import StagedChromaDB         # noqa: E402
from fio.tracking import BuildTracker                  # noqa: E402


# ── Copied from scripts/build_memory.py (unchanged logic) ──────────────────────────────────
def _action_sequence_to_text(action_sequence) -> str:
    if isinstance(action_sequence, list):
        seq = action_sequence
    else:
        if not action_sequence or action_sequence in ("[]", "", "null"):
            return ""
        try:
            seq = json.loads(action_sequence)
        except (json.JSONDecodeError, TypeError, ValueError):
            return ""
    if not isinstance(seq, list):
        return ""
    parts = []
    for step in seq:
        if not isinstance(step, dict):
            continue
        act = (step.get("action") or "").strip()
        obj = (step.get("object") or "").strip()
        sp = (step.get("speech_content") or "").strip()
        if act and obj:
            parts.append(f"{act} {obj}")
        elif act:
            parts.append(act)
        if sp:
            parts.append(f'said: "{sp[:60]}"')
    return ("also: " + "; ".join(parts)) if parts else ""


def build_document_string_opt_aware(fio: Dict) -> str:
    parts = []
    for key, label in (("persons", "persons"), ("action", "action"), ("object", "object"),
                       ("tool", "tool"), ("app", "app"), ("source_location", "from"),
                       ("target_location", "to"), ("speech_content", "speech")):
        val = fio.get(key) or ""
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        val = str(val).strip()
        if val:
            parts.append(f"{label}: {val}")
    extra = (fio.get("opt_document_extra") or "").strip() \
        or _action_sequence_to_text(fio.get("action_sequence", "[]"))
    if extra:
        parts.append(extra)
    attrs = fio.get("attributes", "{}")
    try:
        attrs = json.loads(attrs) if isinstance(attrs, str) else attrs
        if isinstance(attrs, dict):
            parts.extend(f"{k}: {v}" for k, v in attrs.items() if v)
    except (json.JSONDecodeError, TypeError):
        pass
    return " | ".join(parts) if parts else "no data"


def apply_recency_scores(kept: List[Dict], raw_by_day: Dict[int, List[Dict]]) -> None:
    recency = {}
    for day_int, day_raw in raw_by_day.items():
        recency.update(compute_recency_for_day(day_raw, day_int))
    for fio in kept:
        fid = fio.get("id", "")
        if fid in recency:
            fio["recency_score"] = recency[fid]
        imp, rec = fio.get("importance_score", 0.0), fio.get("recency_score", 0.0)
        if rec > 0.0:
            fio["importance_score"] = round(imp * rec, 6)


# ── Build ────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="FIO Phase-2 build with AdaImprint (no LLM)")
    ap.add_argument("--ckpt", default="", help="AdaImprint checkpoint; omit for the heuristic")
    ap.add_argument("--captions", required=True, help="<wearer>_meta.json (caption order)")
    ap.add_argument("--cache-dir", required=True, help="Phase-1 FIO cache directory")
    ap.add_argument("--out", required=True, help="output dir (chroma_fio_opt, build_report.json)")
    ap.add_argument("--target-day", type=int, default=7)
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads for the learned model; it steps one FIO at a time, "
                         "so many threads only add overhead")
    ap.add_argument("--budget", type=int, default=None, help="hard memory budget B (entries)")
    ap.add_argument("--online-lr", type=float, default=0.0, help=">0: learn during the build")
    ap.add_argument("--store-raw", action="store_true", help="also build chroma_fio_raw")
    ap.add_argument("--doc-fix", action="store_true",
                    help="embed merged action_sequence text (fixes fio_v3 known issue 1; "
                         "changes embeddings vs existing builds)")
    ap.add_argument("--local-db-root", default=os.environ.get("LOCAL_DB_ROOT"))
    ap.add_argument("--opt", action="append", default=[],
                    help="override an optimizer knob, e.g. --opt delta_g_threshold=0.7")
    ap.add_argument("--save-ckpt", default="",
                    help="with --online-lr: save the updated model here after the build")
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)

    opt_kwargs = dict(FULL_OPT_KWARGS, track_growth=True)
    for kv in args.opt:
        k, v = kv.split("=", 1)
        if k not in opt_kwargs:
            ap.error(f"unknown optimizer knob {k}")
        t = type(opt_kwargs[k])
        opt_kwargs[k] = (v.lower() in ("1", "true", "yes")) if t is bool else t(v)

    learned = bool(args.ckpt)
    if learned:
        opt = LearnableFIOOptimizer.from_checkpoint(
            args.ckpt, memory_budget=args.budget, online_lr=args.online_lr,
            record_trace=False, quiet=True, **opt_kwargs)
        print(f"[build] AdaImprint {args.ckpt} (budget={args.budget}, online_lr={args.online_lr})")
    else:
        if args.budget or args.online_lr:
            ap.error("--budget/--online-lr need --ckpt")
        # model=None makes exactly FIOOptimizer's decisions (tests/test_smoke.py check 1)
        # without its per-FIO debug output
        opt = LearnableFIOOptimizer(model=None, record_trace=False, quiet=True, **opt_kwargs)
        print("[build] heuristic FIOOptimizer (same as scripts/build_memory.py full)")

    os.makedirs(args.out, exist_ok=True)
    raw_dir = os.path.join(args.out, "chroma_fio_raw")
    opt_dir = os.path.join(args.out, "chroma_fio_opt")
    stager = StagedChromaDB(local_root=args.local_db_root)
    raw_live = stager.register("chroma_fio_raw", raw_dir if args.store_raw else None)
    opt_live = stager.register("chroma_fio_opt", opt_dir)
    storage = DualFIOStorage(raw_db_dir=raw_live or raw_dir, opt_db_dir=opt_live,
                             store_raw=args.store_raw, store_opt=True)

    stream = list(iter_caption_fios(args.captions, args.cache_dir, args.target_day))
    tracker = BuildTracker(model_name="AdaImprint" if learned else "FIOOptimizer",
                           verbose_progress=True, eta_window=20)
    tracker.start_build(total_captions=len(stream))

    all_kept: List[Dict] = []
    raw_by_day: Dict[int, List[Dict]] = {}
    for i, (cap, raw_fios) in enumerate(stream):
        tracker.start_caption(i, day=cap.get("date", "DAY1"))
        tracker.start_gen()
        tracker.end_gen(input_tokens=0, output_tokens=0, success=bool(raw_fios),
                        error_type=None if raw_fios else "empty_cache")
        tracker.start_opt()
        for f in raw_fios:
            f["optimized"] = False
        storage.store_raw_fios(raw_fios, args.target_day)
        day_int = int(re.sub(r"[^0-9]", "", str(cap.get("date", "1"))) or "1")
        raw_by_day.setdefault(day_int, []).extend(raw_fios)

        st = opt._stats
        prev = (st["total_kept"], st["total_dropped"], st["total_compressed"])
        res = opt.process(raw_fios) if raw_fios else {"kept": []}
        new_kept = res["kept"][prev[0]:] if raw_fios else []
        for fio in new_kept:
            fio["optimized"] = True
            fio["recency_score"] = 0.0
        tracker.end_opt(n_raw=len(raw_fios), n_kept=len(new_kept),
                        n_compressed=st["total_compressed"] - prev[2],
                        n_dropped=st["total_dropped"] - prev[1])
        all_kept.extend(new_kept)
        tracker.end_caption()

    if learned:
        # AdaImprint may merge into, update or evict earlier entries: store the final live M.
        all_kept = opt.final_memory()
        for fio in all_kept:
            fio["optimized"] = True
            fio["recency_score"] = 0.0
        print(f"[build] final memory: {opt.memory_report()}")

    apply_recency_scores(all_kept, raw_by_day)
    for fio in all_kept:
        fio["opt_document_extra"] = _action_sequence_to_text(fio.get("action_sequence", "[]"))
    if args.doc_fix:
        # fio.storage.dual and fio.storage.chroma both bind build_document_string by name
        # and fio.storage.chroma rebuilds the document before writing, so patch both modules
        # (opt store only) — this is what fio_v3's monkey-patch meant to do.
        import fio.storage.chroma as chroma_fio
        import fio.storage.dual as chroma_fio_v2
        saved = (chroma_fio.build_document_string, chroma_fio_v2.build_document_string)
        chroma_fio.build_document_string = build_document_string_opt_aware
        chroma_fio_v2.build_document_string = build_document_string_opt_aware
        try:
            storage.store_opt_fios(all_kept, args.target_day)
        finally:
            chroma_fio.build_document_string, chroma_fio_v2.build_document_string = saved
    else:
        storage.store_opt_fios(all_kept, args.target_day)   # same documents as scripts/build_memory.py builds

    tracker.finish_build(optimizer_stats=opt._stats)
    tracker.save_report(os.path.join(args.out, "build_report.json"))
    with open(os.path.join(args.out, "build_config.json"), "w") as f:
        json.dump({"ckpt": args.ckpt, "budget": args.budget, "online_lr": args.online_lr,
                   "doc_fix": args.doc_fix, "opt_kwargs": opt_kwargs,
                   "captions": args.captions, "cache_dir": args.cache_dir,
                   "n_entries": len(all_kept),
                   "memory_report": opt.memory_report() if learned else None}, f, indent=2)
    storage.show_stats()
    stager.sync_all()

    if learned and args.online_lr > 0 and args.save_ckpt:
        from .model import save_checkpoint
        save_checkpoint(args.save_ckpt, opt.lmodel, {"online_from": args.ckpt})
        print(f"[build] online-updated model saved to {args.save_ckpt}")

    from fio.stats import analyze_chroma_dates
    print(f"\n[build] {len(all_kept)} memory entries -> {opt_dir}")
    analyze_chroma_dates(opt_dir)


if __name__ == "__main__":
    main()
