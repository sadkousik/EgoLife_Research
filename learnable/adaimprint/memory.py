"""Differentiable episodic-memory model and the streaming AdaImprint loop.

The episodic memory M is represented as *slots* (content embedding, time,
mass). Each processed FIO adds slots according to its decision:

    write   (e_t,      t_t, m_w)                       new entry with its own evidence
    merge   (e_cand,   t_t, m_m)                       time kept, content is the entry's
    update  (e_t,      t_t, m_u) + (e_t, t_cand, m_u * m_w(cand)),
            and the candidate's own write slot is scaled by (1 - m_u)

with 0/1 masses from the actual decisions; the training loss uses a
straight-through estimator (hard memory forward, gradients through the soft
gate ``g_t`` and the policy ``p(action)``). Merge keeps distinct temporal evidence (the FIO's own time stays in
memory, as in ``action_sequence``) but loses its content; update keeps the new
content and overwrites the old one.

``StreamProcessor`` runs steps 3-8 of the algorithm for one stream:
dynamic features (b_t, E_t, U_t) -> controller -> decision -> memory slots ->
memory-conditioned next-event prediction, and, per chronological chunk,

    L = L_pred + w_past L_past + beta L_mem + gamma L_redund [+ w_ev L_evidence]

where L_past reconstructs randomly sampled past FIOs (within ``horizon``) from
the memory, so that unusual-but-hard-to-predict events are not traded away for
routine, predictable ones.
"""

from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import ACTIONS, LearnableImportance, policy_probs, surprise_from_logits

WRITE, MERGE, UPD_NEW, UPD_MOVED = 0, 1, 2, 3


def readout(E: Tensor, ts: Tensor, m: Tensor, q_ts: Tensor, sigma: float, eps: float,
            q_emb: Optional[Tensor] = None, read_temp: float = 0.1) -> Tensor:
    """r_q = sum_s w_qs E_s / (sum_s w_qs + eps); q_ts [Q] -> [Q, D].

    Time-only read:    w_qs = m_s k(t_q, t_s)
    Content+time read: w_qs = m_s k(t_q, t_s) exp((cos(q, E_s) - 1) / read_temp)

    The content term matters because many FIOs share one clip timestamp, so a
    time-only read cannot tell them apart.
    """
    if E.shape[0] == 0:
        return torch.zeros(q_ts.shape[0], E.shape[1] if E.dim() == 2 else 0)
    w = m[None, :] * torch.exp(-(q_ts[:, None] - ts[None, :]).abs() / sigma)
    if q_emb is not None:
        w = w * torch.exp((q_emb @ E.T - 1.0) / read_temp)
    return (w @ E) / (w.sum(dim=1, keepdim=True) + eps)


def reconstruction(E, ts, m, q_ts, q_proto, centroids, cfg, q_emb=None):
    """Recover the prototype of past FIOs from memory.

    With ``cfg.past_read == "content"`` the query is the FIO's own embedding
    plus its time: the loss is low only if memory still holds a slot that is
    close to the FIO in both content and time (what RAG retrieval needs).
    """
    content = getattr(cfg, "past_read", "time") == "content" and q_emb is not None
    r = readout(E, ts, m, q_ts, cfg.sigma, cfg.read_eps,
                q_emb=q_emb if content else None, read_temp=getattr(cfg, "read_temp", 0.1))
    logits = r @ centroids.T / cfg.cls_temp
    return F.cross_entropy(logits, q_proto), (logits.argmax(1) == q_proto).float().mean()


def build_slots(g, P, cand, tab_emb, tab_ts, n_buf, buf=None):
    """Vectorised slot construction for the last L rows of an item table.

    g [L], P [L,3] (write/merge/update, rows without a candidate = [1,0,0]),
    cand [L] index into the table or -1. ``buf`` holds detached slots of the
    first n_buf items: E, ts, m, write_idx [n_buf] (slot index or -1).
    """
    L = g.shape[0]
    e, ts = tab_emb[n_buf:], tab_ts[n_buf:]
    M = g[:, None] * P
    mw, mm, mu = M[:, 0], M[:, 1], M[:, 2]
    has = cand >= 0
    cc = cand.clamp_min(0)
    log_keep = torch.log1p(-mu.clamp(max=1 - 1e-6))
    in_chunk = has & (cand >= n_buf)
    in_buf = has & (cand < n_buf)
    ci = (cand - n_buf).clamp(0, max(L - 1, 0))
    scale_c = torch.zeros(L).index_add(0, ci[in_chunk], log_keep[in_chunk]).exp()
    cand_mw = torch.where(in_chunk, mw[ci], torch.zeros(L))

    Es, Ts, Ms = [e * 1.0, tab_emb[cc], e, e], [ts, ts, ts, tab_ts[cc]], [mw * scale_c, mm * has, mu * has]
    if buf is not None and buf["E"].shape[0] > 0:
        widx = torch.where(in_buf, buf["write_idx"][cand.clamp(0, max(n_buf - 1, 0))],
                           torch.full((L,), -1, dtype=torch.long))
        ok = widx >= 0
        scale_b = torch.zeros(buf["m"].shape[0]).index_add(0, widx[ok], log_keep[ok]).exp()
        cand_mw = cand_mw + torch.where(ok, buf["m"][widx.clamp_min(0)], torch.zeros(L))
        Es.append(buf["E"]); Ts.append(buf["ts"])
        buf_m = buf["m"] * scale_b
    else:
        buf_m = None
    Ms.append(mu * cand_mw * has)
    if buf_m is not None:
        Ms.append(buf_m)
    return torch.cat(Es), torch.cat(Ts), torch.cat(Ms)


class _Growing:
    """Append-only tensor columns with a moving start (for horizon trimming)."""

    def __init__(self, specs: Dict[str, tuple]):
        self.specs = specs
        self.cap = 1024
        self.cols = {k: torch.zeros((self.cap,) + shp, dtype=dt) for k, (shp, dt) in specs.items()}
        self.n = 0
        self.start = 0
        self.base = 0          # absolute index of row 0

    def append(self, **vals) -> int:
        if self.n == self.cap:
            if self.start > self.cap // 2:
                for k in self.cols:
                    self.cols[k][: self.n - self.start] = self.cols[k][self.start:self.n].clone()
                self.base += self.start
                self.n -= self.start
                self.start = 0
            else:
                self.cap *= 2
                for k, (shp, dt) in self.specs.items():
                    new = torch.zeros((self.cap,) + shp, dtype=dt)
                    new[: self.n] = self.cols[k][: self.n]
                    self.cols[k] = new
        for k, v in vals.items():
            self.cols[k][self.n] = v
        self.n += 1
        return self.base + self.n - 1

    def row(self, abs_idx: int) -> int:
        return abs_idx - self.base

    def active(self, k: str) -> Tensor:
        return self.cols[k][self.start:self.n]


class SlotRing:
    def __init__(self, dim: int):
        self.g = _Growing({"E": ((dim,), torch.float32), "ts": ((), torch.float32),
                           "m": ((), torch.float32), "origin": ((), torch.long),
                           "kind": ((), torch.long)})
        self.write_slot: Dict[int, int] = {}

    def add(self, e, ts, m, origin, kind):
        if kind != WRITE and m <= 0.0:
            return
        idx = self.g.append(E=e, ts=ts, m=m, origin=origin, kind=kind)
        if kind == WRITE:
            self.write_slot[origin] = idx

    def _slot_row(self, origin: int) -> Optional[int]:
        idx = self.write_slot.get(origin)
        if idx is None:
            return None
        r = self.g.row(idx)
        return r if r >= self.g.start else None

    def write_mass(self, origin: int) -> float:
        r = self._slot_row(origin)
        return float(self.g.cols["m"][r]) if r is not None else 0.0

    def scale_write(self, origin: int, factor: float) -> None:
        r = self._slot_row(origin)
        if r is not None:
            self.g.cols["m"][r] *= factor

    def window(self, min_origin: int):
        o = self.g.active("origin")
        i = int(torch.searchsorted(o, torch.tensor(min_origin))) + self.g.start
        sl = slice(i, self.g.n)
        c = self.g.cols
        return c["E"][sl], c["ts"][sl], c["m"][sl], c["origin"][sl], c["kind"][sl]

    def trim(self, min_origin: int) -> None:
        o = self.g.active("origin")
        self.g.start += int(torch.searchsorted(o, torch.tensor(min_origin)))
        for k in [k for k, v in self.write_slot.items() if k < min_origin]:
            del self.write_slot[k]


class ItemRing:
    def __init__(self, dim: int):
        self.g = _Growing({"emb": ((dim,), torch.float32), "ts": ((), torch.float32),
                           "proto": ((), torch.long), "kept": ((), torch.bool)})

    def add(self, origin, emb, ts, proto, kept):
        idx = self.g.append(emb=emb, ts=ts, proto=proto, kept=kept)
        assert idx == origin, (idx, origin)

    @property
    def first(self) -> int:
        return self.g.base + self.g.start

    def has(self, origin: int) -> bool:
        return self.first <= origin < self.g.base + self.g.n

    def get(self, k: str, origin: int):
        return self.g.cols[k][self.g.row(origin)]

    def range(self, k: str, o0: int, o1: int) -> Tensor:
        return self.g.cols[k][self.g.row(o0):self.g.row(o1)]

    def trim(self, min_origin: int) -> None:
        self.g.start = max(self.g.start, self.g.row(min_origin))


class StreamProcessor:
    """Steps 3-8 of AdaImprint for one stream (training, evaluation or build)."""

    def __init__(self, model: LearnableImportance, centroids_fn: Callable[[], Tensor], *,
                 budget: float, training: bool, keep_all: bool = False, ts0: float = 0.0):
        self.model, self.cfg = model, model.cfg
        self.centroids_fn = centroids_fn
        self.budget = max(1.0, float(budget))
        self.training = training
        self.keep_all = keep_all          # never trim (evaluation over the whole stream)
        self.ts0 = ts0                    # subtracted from timestamps (float32 safety)
        self.ring = SlotRing(self.cfg.emb_dim)
        self.items = ItemRing(self.cfg.emb_dim)
        self.t = 0
        self.state = None
        self.prev_logits = None
        self.pending = None
        self.writes = 0.0
        self.next_hits = 0
        self.next_n = 0
        self.counts = {"discard": 0, "write": 0, "merge": 0, "update": 0}
        self._begin_chunk()

    # ------------------------------------------------------------------
    def step(self, raw: Tensor, emb: Tensor, ts: float, proto: int, cand: int, *,
             cand_sim: float = 1.0, occupancy: Optional[float] = None, cand_valid: Optional[bool] = None,
             decide: Optional[Callable] = None, evidence: bool = False) -> Dict:
        cfg, model, t = self.cfg, self.model, self.t
        ts = float(ts) - self.ts0
        if self.prev_logits is not None:
            self.next_hits += int(self.prev_logits.argmax() == proto)
            self.next_n += 1
        if self.pending is not None:
            self.next_pairs.append((self.pending, proto))

        occ = self.writes / self.budget if occupancy is None else occupancy
        sur = surprise_from_logits(self.prev_logits, proto, cfg.n_protos)
        E_w, _, m_w, _, _ = self.ring.window(t - cfg.cand_window)
        U = float(((E_w @ emb) * m_w.clamp(max=1.0)).max()) if E_w.shape[0] else 0.0
        dyn = torch.tensor([[occ, sur, U]], dtype=torch.float32)
        out = model.step(raw[None], emb[None], dyn, self.state)

        # cand_valid: a same-pattern entry is in memory (similarity >= cons_floor);
        # whether it is close enough to merge into is the learned consolidation gate
        if cand_valid is None:
            cand_valid = cand >= 0 and self.items.has(cand) and bool(self.items.get("kept", cand))
        if not cand_valid or not self.items.has(cand):
            cand = -1                    # no equivalent entry, or its content is outside the ring
            cand_valid = False
        q = model.merge_eligibility(cand_sim) if cand_valid else None
        eligible = cand_valid and model.merge_eligible(cand_sim)
        P = policy_probs(out["policy_logits"][0], cand_valid, q)
        keep = bool(out["log_w"][0] >= model.log_tau)
        action = int(out["policy_logits"][0].argmax()) if eligible else 0
        if decide is not None:
            keep, action = decide(keep, action, out)

        g = out["gate"][0]
        masses = torch.zeros(3)          # the memory holds actual (hard) decisions
        if keep:
            masses[action] = 1.0
        mw, mm, mu = (float(x) for x in masses)
        self.counts["discard" if not keep else ACTIONS[action]] += 1

        self.ring.add(emb, ts, mw, t, WRITE)
        if cand >= 0:
            c_emb, c_ts = self.items.get("emb", cand), float(self.items.get("ts", cand))
            self.ring.add(c_emb, ts, mm, t, MERGE)
            if mu > 0:
                self.ring.add(emb, ts, mu, t, UPD_NEW)
                self.ring.add(emb, c_ts, mu * self.ring.write_mass(cand), t, UPD_MOVED)
                self.ring.scale_write(cand, 1.0 - mu)
        self.items.add(t, emb, ts, proto, keep)
        self.writes += mw

        if model.next_head is not None:
            E_p, T_p, M_p, _, _ = self.ring.window(t + 1 - cfg.pred_window)
            r = readout(E_p, T_p, M_p, torch.tensor([ts]), cfg.sigma, cfg.read_eps)
            logits = model.predict_next(out["h"], r)[0]
            self.prev_logits = logits.detach()
            self.pending = logits if self.training else None
        self.state = out.get("state")

        self.chunk.append({"g": g, "P": P, "cand": cand, "keep": keep, "action": action,
                           "evidence": evidence})
        self.t += 1
        return {"keep": keep, "action": ACTIONS[action], "eligible": eligible,
                "log_w": float(out["log_w"][0].detach()),
                "gate": float(g.detach()), "probs": P.detach()}

    # ------------------------------------------------------------------
    def _begin_chunk(self) -> None:
        self.chunk: List[Dict] = []
        self.next_pairs: List = []
        self.chunk_start = self.t
        self.q0 = max(self.items.first, self.t - self.cfg.horizon) if self.t else 0
        n_buf = self.t - self.q0
        E, T, M, O, K = self.ring.window(self.q0)
        write_idx = torch.full((n_buf,), -1, dtype=torch.long)
        sel = (K == WRITE) & (O < self.t)
        rows = torch.nonzero(sel).squeeze(1)
        write_idx[O[rows] - self.q0] = rows
        self.buf = {"E": E.clone(), "ts": T.clone(), "m": M.clone(), "write_idx": write_idx}

    def chunk_loss(self, rng: Optional[torch.Generator] = None, *, w_past=1.0, beta=5.0,
                   gamma=0.1, w_evidence=0.0, rate_loss="two_sided", merge_cost=0.05,
                   past_queries=64, redund_band=16, redund_delta=0.75) -> Dict[str, Tensor]:
        cfg, recs = self.cfg, self.chunk
        n_buf = self.chunk_start - self.q0
        g_soft = torch.stack([r["g"] for r in recs])
        P_soft = torch.stack([r["P"] for r in recs])
        cand = torch.tensor([r["cand"] - self.q0 if r["cand"] >= self.q0 else -1 for r in recs])
        P_soft = torch.where((cand >= 0)[:, None], P_soft, torch.tensor([1.0, 0.0, 0.0]))
        # straight-through: the loss sees the memory the stream actually built,
        # gradients flow through the soft gate and policy
        keep = torch.tensor([float(r["keep"]) for r in recs])
        act = F.one_hot(torch.tensor([r["action"] for r in recs]), 3).float()
        g = keep + g_soft - g_soft.detach()
        P = act + P_soft - P_soft.detach()
        tab_emb = self.items.range("emb", self.q0, self.t)
        tab_ts = self.items.range("ts", self.q0, self.t)
        tab_proto = self.items.range("proto", self.q0, self.t)
        E, T, M = build_slots(g, P, cand, tab_emb, tab_ts, n_buf, self.buf)

        q = torch.randint(0, tab_ts.shape[0], (past_queries,), generator=rng)
        C = self.centroids_fn()
        L_past, _ = reconstruction(E, T, M, tab_ts[q], tab_proto[q], C, cfg, q_emb=tab_emb[q])

        if self.next_pairs:
            NL = torch.stack([p[0] for p in self.next_pairs])
            NT = torch.tensor([p[1] for p in self.next_pairs])
            L_pred = F.cross_entropy(NL, NT)
        else:
            L_pred = g.sum() * 0

        write_m = g * P[:, 0]
        dev = write_m.mean() - cfg.write_rate
        L_mem = (dev.abs() if rate_loss == "two_sided" else F.relu(dev)) \
            + merge_cost * (g * (P[:, 1] + P[:, 2])).mean()

        e = tab_emb[n_buf:]
        L = e.shape[0]
        if L > 1:
            i = torch.arange(L)
            d = i[None, :] - i[:, None]
            band = (d > 0) & (d <= redund_band)
            pen = write_m[:, None] * write_m[None, :] * F.relu(e @ e.T - redund_delta) * band
            L_red = pen.sum() / band.sum().clamp_min(1)
        else:
            L_red = g.sum() * 0

        loss = L_pred + w_past * L_past + beta * L_mem + gamma * L_red
        L_ev = g.sum() * 0
        pos = torch.tensor([r["evidence"] for r in recs])
        if w_evidence > 0 and pos.any():
            L_ev = -torch.log(g_soft[pos] + 1e-6).mean()
            loss = loss + w_evidence * L_ev
        return {"loss": loss, "pred": L_pred, "past": L_past, "mem": L_mem,
                "redund": L_red, "evid": L_ev, "write_rate": write_m.mean().detach()}

    def end_chunk(self) -> None:
        if self.state is not None:
            self.state = tuple(x.detach() for x in self.state)
        self.pending = None
        if not self.keep_all:
            keep_from = self.t - self.cfg.horizon - self.cfg.cand_window
            self.ring.trim(keep_from)
            self.items.trim(max(self.items.first, keep_from))
        self._begin_chunk()

    # ------------------------------------------------------------------
    def memory(self):
        E, T, M, _, _ = self.ring.window(self.items.first)
        return E, T, M

    def stats(self) -> Dict[str, float]:
        n = max(1, self.t)
        out = {f"{k}_rate": v / n for k, v in self.counts.items()}
        if self.next_n:
            out["next_acc"] = self.next_hits / self.next_n
        return out
