"""Train AdaImprint on recorded traces (offline, chronological, truncated BPTT).

    python -m learnable.scripts.train \
        --train learnable/outputs/traces/A2_ALICE.pt ... \
        --val   learnable/outputs/traces/A1_JAKE.pt \
        --out   learnable/outputs/ckpt_full

Per chunk of ``--chunk`` FIOs (LSTM state carried across chunks, graph cut):

    L = L_pred + w_past * L_past + beta * L_mem + gamma * L_redund [+ w_ev * L_evidence]

* L_pred    next-prototype prediction from [h_t, memory read-out]  (P_omega)
* L_past    reconstruct randomly sampled past FIOs (last --horizon FIOs) from
            the soft memory: protects rare, hard-to-predict events that pure
            next-event prediction would let the policy discard
* L_mem     |write rate - r*| (or hinge) + small cost for merge/update
* L_redund  writing near-duplicate neighbours
* L_evidence optional, EgoLifeQA target_time of *training* wearers only

With one wearer, --split-day N trains on DAY1..DAY(N-1) and validates on
DAYN onwards of the same trace.

r* (``--write-rate``) defaults to the heuristic's new-entry rate on the training
traces, so learned and heuristic memories are compared at equal size.
--mode params learns only the four scalars of the importance formula.
"""

import argparse
import json
import os
import random
import time

import torch

from learnable.adaimprint.data import attach_evidence, load_trace, split_by_day
from learnable.adaimprint.model import (
    DYNAMIC_INPUTS, FEATURE_NAMES, LearnableImportance, LearnableImportanceConfig,
    save_checkpoint,
)
from learnable.adaimprint.training import evaluate, fmt, run_trace


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train AdaImprint (learnable FIO memory)")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--split-day", type=int, default=0,
                    help="single wearer: train on DAY1..DAY(n-1) of each --train trace, "
                         "validate on DAYn onwards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("full", "params"), default="full")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=2048, help="past FIOs eligible for L_past")
    ap.add_argument("--past-queries", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=300.0, help="temporal kernel scale (s)")
    ap.add_argument("--cls-temp", type=float, default=0.1)
    ap.add_argument("--past-read", choices=("content", "time"), default="content",
                    help="L_past read-out: query memory by content+time (default) or time only")
    ap.add_argument("--read-temp", type=float, default=0.1,
                    help="content-similarity temperature of the content read")
    ap.add_argument("--pred-window", type=int, default=256)
    ap.add_argument("--write-rate", type=float, default=None,
                    help="target new entries per FIO; default: heuristic's rate on --train")
    ap.add_argument("--w-past", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=5.0, help="L_mem weight")
    ap.add_argument("--rate-loss", choices=("two_sided", "hinge"), default="two_sided")
    ap.add_argument("--merge-cost", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.1, help="L_redund weight")
    ap.add_argument("--redund-band", type=int, default=16)
    ap.add_argument("--redund-delta", type=float, default=0.75)
    ap.add_argument("--evidence", nargs="*", default=[],
                    help="TRACE_PATH=EgoLifeQA.json pairs (training wearers only)")
    ap.add_argument("--evidence-window", type=float, default=30.0)
    ap.add_argument("--w-evidence", type=float, default=0.5)
    ap.add_argument("--cons-temp", type=float, default=0.05,
                    help="temperature of the soft consolidation gate")
    ap.add_argument("--fix-cons", action="store_true",
                    help="keep the consolidation threshold at the heuristic's value")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--gate-temp", type=float, default=0.25)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU threads. The stream is processed one FIO at a time with tiny "
                         "tensors, so more threads (or a GPU) make training slower, not faster")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    train = [load_trace(p) for p in args.train]
    val = [load_trace(p) for p in args.val]
    if args.split_day:
        # single-wearer mode: earlier days train, later days validate (per trace)
        if args.evidence:
            ap.error("--evidence cannot be combined with --split-day")
        pairs = [split_by_day(t, args.split_day) for t in train]
        train = [a for a, _ in pairs]
        val = val + [b for _, b in pairs]
        for a, b in pairs:
            print(f"[split] {a['name']}: {a['feats'].shape[0]} FIOs | {b['name']}: {b['feats'].shape[0]} FIOs")
    for pair in args.evidence:
        tp, qp = pair.split("=", 1)
        if any(os.path.abspath(tp) == os.path.abspath(v["path"]) for v in val):
            ap.error(f"evidence given for validation trace {tp} — that leaks the labels")
        match = [t for t in train if os.path.abspath(t["path"]) == os.path.abspath(tp)]
        if not match:
            ap.error(f"--evidence trace {tp} is not in --train")
        attach_evidence(match[0], qp, args.evidence_window)
    if not val:
        print("[warn] no --val traces; reporting metrics on the training traces")
        val = train

    cons = {t["cons_thr"] for t in train + val}
    if len(cons) > 1:
        ap.error(f"traces were recorded with different compression_sim_threshold: {cons}")
    cons_thr = cons.pop()
    cons_floor = max(t["cand_floor"] for t in train + val)
    taus = {t["tau"] for t in train + val if t["tau"] is not None}
    lams = {t["lambda"] for t in train + val if t["lambda"] is not None}
    if len(taus) > 1 or len(lams) > 1:
        ap.error(f"traces were recorded with different tau/lambda: {taus} {lams}")
    tau = taus.pop() if taus else 0.2
    lam = lams.pop() if lams else 1.146e-6
    n_protos = max(max(t["max_prototypes"] for t in train + val),
                   max(int(t["proto"].max()) + 1 for t in train + val))
    heur_rate = sum(t["heur_write_rate"] for t in train) / len(train)
    write_rate = args.write_rate if args.write_rate is not None else heur_rate

    cfg = LearnableImportanceConfig(
        emb_dim=train[0]["emb"].shape[1], hidden_dim=args.hidden, n_protos=n_protos,
        tau=tau, lambda_init=lam, gate_temp=args.gate_temp, use_controller=(args.mode == "full"),
        write_rate=write_rate,
        expected_stream_len=int(sum(t["feats"].shape[0] for t in train) / len(train)),
        sigma=args.sigma, cls_temp=args.cls_temp, pred_window=args.pred_window,
        past_read=args.past_read, read_temp=args.read_temp,
        cons_thr_init=cons_thr, cons_temp=args.cons_temp, cons_floor=cons_floor,
        learn_cons=not args.fix_cons,
        horizon=args.horizon,
    )
    model = LearnableImportance(cfg)
    model.norm.update(torch.cat([t["feats"] for t in train]))

    with torch.no_grad():
        for t in train:
            err = (model.base_log_w(t["feats"])
                   - t["feats"][:, FEATURE_NAMES.index("log_base_w")]).abs().max()
            if err > 1e-3:
                raise RuntimeError(f"{t['name']}: parametric init differs from heuristic by {err:.2e}")
    print(f"[init] {len(train)} train / {len(val)} val traces, "
          f"{sum(t['feats'].shape[0] for t in train)} train FIOs, tau={tau}, lambda={lam:.3e}, "
          f"target write rate={write_rate:.3f} (heuristic {heur_rate:.3f}), mode={args.mode}, "
          f"inputs={len(FEATURE_NAMES)}+{len(DYNAMIC_INPUTS)}, "
          f"consolidation threshold={cons_thr} (floor {cons_floor}, "
          f"{'fixed' if args.fix_cons else 'learned'})")

    def sel(m):
        return m["recon_ce"] + args.beta * max(0.0, abs(m["write_rate"] - write_rate) - 0.02)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    m0 = evaluate(model, val, args)
    print(f"[epoch 0 = heuristic init] {fmt(m0)}")
    history = [{"epoch": 0, "val": m0, "params": model.summary()}]
    best = sel(m0)
    save_checkpoint(os.path.join(args.out, "best.pt"), model, {"epoch": 0, "val": m0, "args": vars(args)})

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        order = list(range(len(train)))
        random.shuffle(order)
        logs = []
        for i in order:
            r = run_trace(model, train[i], args, optimizer=opt, rng=rng)
            r.pop("_proc")
            logs.append(r)
        trm = {k: sum(r[k] for r in logs) / len(logs) for k in logs[0]}
        vm = evaluate(model, val, args)
        print(f"[epoch {ep}] {time.time() - t0:.0f}s train {fmt(trm)} | val {fmt(vm)} | {model.summary()}")
        history.append({"epoch": ep, "train": trm, "val": vm, "params": model.summary()})
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        save_checkpoint(os.path.join(args.out, "last.pt"), model, {"epoch": ep, "val": vm, "args": vars(args)})
        if sel(vm) < best:
            best = sel(vm)
            save_checkpoint(os.path.join(args.out, "best.pt"), model,
                            {"epoch": ep, "val": vm, "args": vars(args)})
    print(f"[done] best checkpoint: {os.path.join(args.out, 'best.pt')}")


if __name__ == "__main__":
    main()
