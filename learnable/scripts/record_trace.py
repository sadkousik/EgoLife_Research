"""Record a training trace by replaying the Phase-1 FIO cache through the
heuristic optimizer (no LLM, CPU only).

    cd <repo root>
    python -m learnable.scripts.record_trace \
        --captions generated_caption/A1_JAKE_meta.json \
        --cache-dir /path/to/A1_JAKE/fio_cache \
        --out learnable/outputs/traces/A1_JAKE.pt

The trace holds, per FIO in stream order: the feature row, the contextual
embedding, the clip timestamp, the prototype id, and the heuristic w(f_t).
Features are decision-independent, so one trace per wearer serves every
training run that keeps the same clustering knobs (delta_g, delta_h, beta, ...).
"""

import argparse
import os
import time

from learnable.adaimprint.data import iter_caption_fios
from learnable.adaimprint.fio_adapter import FULL_OPT_KWARGS, LearnableFIOOptimizer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--captions", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-day", type=int, default=7)
    ap.add_argument("--wearer", default="")
    ap.add_argument("--cand-floor", type=float, default=0.5,
                    help="record consolidation candidates down to this similarity "
                         "(the learned threshold cannot go below it)")
    ap.add_argument("--opt", action="append", default=[],
                    help="override an optimizer knob, e.g. --opt delta_g_threshold=0.7")
    args = ap.parse_args(argv)

    opt_kwargs = dict(FULL_OPT_KWARGS)
    for kv in args.opt:
        k, v = kv.split("=", 1)
        if k not in opt_kwargs:
            ap.error(f"unknown optimizer knob {k}")
        t = type(opt_kwargs[k])
        opt_kwargs[k] = (v.lower() in ("1", "true", "yes")) if t is bool else t(v)

    opt = LearnableFIOOptimizer(model=None, quiet=True, cand_floor=args.cand_floor, **opt_kwargs)
    t0, n_caps = time.time(), 0
    for _cap, fios in iter_caption_fios(args.captions, args.cache_dir, args.target_day):
        if fios:
            opt.process(fios)
        n_caps += 1
        if n_caps % 500 == 0:
            print(f"[trace] {n_caps} captions, {len(opt.trace['feats'])} FIOs, "
                  f"{time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    blob = opt.export_trace(args.out, meta={
        "wearer": args.wearer, "captions": args.captions, "cache_dir": args.cache_dir,
        "target_day": args.target_day, "opt_kwargs": opt_kwargs,
    })
    st = blob["stats"]
    retained = st["total_kept"] + st["total_compressed"]
    print(f"[trace] wrote {args.out}: {blob['feats'].shape[0]} FIOs, "
          f"{blob['centroids'].shape[0]} prototypes, heuristic retain rate "
          f"{retained / max(1, st['total_input']):.3f} (kept {st['total_kept']}, "
          f"consolidated {st['total_compressed']}, dropped {st['total_dropped']})")


if __name__ == "__main__":
    main()
