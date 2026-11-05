"""Streaming a trace through the model: training epochs and evaluation.

Per chunk of ``args.chunk`` FIOs (LSTM state carried across chunks, graph cut):

    L = L_pred + w_past * L_past + beta * L_mem + gamma * L_redund [+ w_ev * L_evidence]

Evaluation compares the learned memory with the heuristic at its own
threshold and with the heuristic moved to the same memory size.
"""

import math
from typing import Dict

import torch

from .data import heuristic_valid, padded_centroids
from .memory import StreamProcessor, build_slots, reconstruction

LOSS_KW = ("w_past", "beta", "gamma", "w_evidence", "rate_loss", "merge_cost",
           "past_queries", "redund_band", "redund_delta")


def run_trace(model, tr, args, optimizer=None, rng=None) -> Dict:
    train = optimizer is not None
    cfg = model.cfg
    T = tr["feats"].shape[0]
    C = padded_centroids(tr, cfg.n_protos)
    proc = StreamProcessor(model, lambda: C, budget=cfg.write_rate * T,
                           training=train, keep_all=not train)
    ev = tr["evidence"]
    kw = {k: getattr(args, k) for k in LOSS_KW}
    agg, n = {}, 0
    for t in range(T):
        proc.step(tr["feats"][t], tr["emb"][t], float(tr["ts"][t]), int(tr["proto"][t]),
                  int(tr["cand"][t]), cand_sim=float(tr["cand_sim"][t]),
                  evidence=bool(ev[t]) if ev is not None else False)
        if train and ((t + 1) % args.chunk == 0 or t == T - 1):
            losses = proc.chunk_loss(rng, **kw)
            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            n += 1
            proc.end_chunk()
    res = {k: v / max(1, n) for k, v in agg.items()}
    res.update(proc.stats())
    res["_proc"] = proc
    return res


@torch.no_grad()
def recon_metrics(tr, E, T_, M, cfg, n_queries=2000, seed=0) -> Dict[str, float]:
    g = torch.Generator().manual_seed(seed)
    N = tr["ts"].shape[0]
    q = torch.randperm(N, generator=g)[: min(n_queries, N)]
    C = padded_centroids(tr, cfg.n_protos)
    ce = acc = 0.0
    for qb in q.split(256):
        c, a = reconstruction(E, T_, M, tr["ts"][qb], tr["proto"][qb], C, cfg, q_emb=tr["emb"][qb])
        ce += float(c) * len(qb)
        acc += float(a) * len(qb)
    return {"recon_ce": ce / len(q), "recon_acc": acc / len(q)}


@torch.no_grad()
def heuristic_memory(tr, thr: float):
    keep = tr["base_w"] >= thr
    cand = tr["cand"]
    valid = heuristic_valid(cand, tr["cand_sim"], keep, tr["cons_thr"])
    P = torch.zeros(len(keep), 3)
    P[torch.arange(len(keep)), torch.where(valid, 1, 0)] = 1.0
    E, T_, M = build_slots(keep.float(), P, torch.where(valid, cand, -1), tr["emb"], tr["ts"], 0)
    return (E, T_, M), int((keep & ~valid).sum())


@torch.no_grad()
def evaluate(model, traces, args) -> Dict[str, float]:
    model.eval()
    rows = []
    for tr in traces:
        r = run_trace(model, tr, args)
        proc = r.pop("_proc")
        N = tr["ts"].shape[0]
        row = {k: v for k, v in r.items()}
        row.update(recon_metrics(tr, *proc.memory(), model.cfg))
        (Eh, Th, Mh), w_tau = heuristic_memory(tr, model.cfg.tau)
        h = recon_metrics(tr, Eh, Th, Mh, model.cfg)
        row.update({"heur_write_rate": w_tau / N, "heur_recon_acc": h["recon_acc"],
                    "heur_recon_ce": h["recon_ce"]})
        # heuristic with its threshold moved so it writes as many entries as the learned
        # model (grid scan: the write count is not monotone in the threshold, because
        # dropping an entry turns its would-be merges into writes)
        target_writes = round(r["write_rate"] * N)
        grid = torch.unique(torch.quantile(tr["base_w"], torch.linspace(0, 1, 201)))
        grid = torch.cat([grid, torch.tensor([model.cfg.tau])])
        best_thr, best_key = model.cfg.tau, None
        for thr in grid.tolist():
            _, w = heuristic_memory(tr, thr)
            key = (abs(w - target_writes), abs(math.log(max(thr, 1e-12) / model.cfg.tau)))
            if best_key is None or key < best_key:
                best_thr, best_key = thr, key
        hi = best_thr
        (Em, Tm, Mm), _ = heuristic_memory(tr, hi)
        hm = recon_metrics(tr, Em, Tm, Mm, model.cfg)
        row.update({"heur_matched_recon_acc": hm["recon_acc"], "heur_matched_recon_ce": hm["recon_ce"]})
        rows.append(row)
    model.train()
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def fmt(d):
    return " ".join(f"{k}={v:.4f}" for k, v in d.items())
