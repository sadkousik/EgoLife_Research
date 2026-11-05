"""Learnable FIO importance, write policy and memory-conditioned predictor (AdaImprint).

Importance (step 3 + 5 of the algorithm)
----------------------------------------
The heuristic of ``FIOOptimizer``

    I_t = log(1 + nu) * exp(-lambda dT) / (1 + log(1 + N))          (F_t * R_t * D_t)

becomes

    log I_adapt = a_freq log(log(1+nu)) - lambda dT - a_div log(1+log(1+N)) + b + delta_t

with ``delta_t = CorrectionHead(h_t, z_t)`` bounded to [-2, 2]. The soft
retention gate is ``g_t = sigmoid((log I_adapt - log tau) / T)``; at inference
an FIO is retained iff ``I_adapt >= tau``.

Policy (step 5/6)
-----------------
p(discard) = 1 - g_t. A retained FIO that has an equivalent memory entry
chooses between write (new entry, keeps its own evidence), merge (fold into the
entry's action_sequence) and update (fold in and make it the entry's new
canonical content): p(write|merge|update) = g_t * softmax(policy_logits).
Without an equivalent entry the FIO can only be written.

Predictor P_omega (step 4/8)
---------------------------
next-prototype logits = Linear([h_t, r_t]) where r_t is a temporal-kernel
read-out of the compressed memory, so the prediction is made from what the
memory actually holds.

Initialisation reproduces the heuristic exactly: a = 1, b = 0, lambda = build
lambda, zero correction head, and a policy bias that always picks merge (the
heuristic consolidates whenever an equivalent entry exists).
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .controller import AdaImprintConfig, AdaImprintController, RunningFeatureNorm

# Static per-FIO features recorded by LearnableFIOOptimizer. They depend only on
# the optimizer's decision-independent state (prototype bank G, its statistics,
# the history buffer), so a trace recorded with the heuristic is what the
# learned model sees at build time whatever it decides to keep.
FEATURE_NAMES: Tuple[str, ...] = (
    "log1p_freq",         # F_t: log(1 + nu), nu as used by compute_weight (>= 1)
    "dt_sec",             # Delta_t: effective seconds since the prototype was last seen
    "log1p_n_protos",     # D_t: log(1 + N), N = active prototypes
    "log_base_w",         # log I_t (heuristic)
    "novelty",            # S_t: 1 - best prototype similarity before assignment
    "is_new_proto",       # 1 if this FIO spawned a new prototype
    "hist_max_sim",       # max cos(e_t, e_j) over the history buffer
    "hist_related_frac",  # |{j : cos(e_t, e_j) > delta_h}| / history_window
    "field_richness",
    "has_speech",
    "n_persons",
    "has_tool_or_app",
    "tod_sin",
    "tod_cos",
    "day_frac",
    "clip_pos",
)
F_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
# Memory-dependent inputs, computed while streaming (training loop or build).
DYNAMIC_INPUTS = (
    "occupancy",        # b_t = |M| / B
    "surprise",         # E_t = -log P_omega(proto_t | h_{t-1}, M_{t-1}) / log(P)
    "mem_redundancy",   # U_t = max_s cos(e_t, slot_s) * min(1, mass_s) over recent memory
)
ACTIONS = ("write", "merge", "update")


@dataclass
class LearnableImportanceConfig:
    n_features: int = len(FEATURE_NAMES)
    emb_dim: int = 384
    emb_proj_dim: int = 16
    hidden_dim: int = 64
    n_protos: int = 300
    tau: float = 0.2
    lambda_init: float = 1.146e-6
    gate_temp: float = 0.25
    max_log_correction: float = 2.0
    use_controller: bool = True
    # memory / budget
    write_rate: float = 0.15        # target new entries per input FIO (training budget)
    expected_stream_len: int = 10000  # B = write_rate * this when no hard budget is given
    sigma: float = 300.0            # temporal kernel scale (s) for memory read-out
    cls_temp: float = 0.1
    past_read: str = "content"      # L_past / recon read-out: "content" (time+content) or "time"
    read_temp: float = 0.1          # content-similarity temperature for the content read
    # consolidation gate: a same-pattern memory entry is a merge/update target iff
    # cos(ctx_t, ctx_entry) >= cons_thr (heuristic: compression_sim_threshold = 0.75)
    cons_thr_init: float = 0.75
    cons_temp: float = 0.05         # soft gate sigma((sim - cons_thr) / cons_temp)
    cons_floor: float = 0.5         # traces record candidates down to this similarity
    learn_cons: bool = True
    read_eps: float = 1e-3
    pred_window: int = 256          # predictor reads slots from the last N FIOs
    cand_window: int = 100          # equivalence / redundancy window (= max_kept_for_compression)
    cand_sim: float = 0.75          # = compression_sim_threshold
    horizon: int = 2048             # past-memory horizon for L_past (FIOs)
    feature_names: Sequence[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    dynamic_inputs: Sequence[str] = field(default_factory=lambda: list(DYNAMIC_INPUTS))

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["feature_names"] = list(d["feature_names"])
        d["dynamic_inputs"] = list(d["dynamic_inputs"])
        return d


_MERGE_BIAS = 4.0  # init: argmax = merge, as in the heuristic


class LearnableImportance(nn.Module):
    def __init__(self, cfg: LearnableImportanceConfig):
        super().__init__()
        if list(cfg.feature_names) != list(FEATURE_NAMES) or \
                list(cfg.dynamic_inputs) != list(DYNAMIC_INPUTS):
            raise ValueError("feature set differs from this code version; re-record traces "
                             "and retrain")
        self.cfg = cfg
        self.a_freq = nn.Parameter(torch.tensor(1.0))
        self.a_div = nn.Parameter(torch.tensor(1.0))
        self.log_lambda = nn.Parameter(torch.tensor(math.log(cfg.lambda_init)))
        self.bias = nn.Parameter(torch.tensor(0.0))
        # learnable consolidation threshold (FIO compression_sim_threshold)
        self.cons_thr = nn.Parameter(torch.tensor(float(cfg.cons_thr_init)),
                                     requires_grad=bool(cfg.learn_cons))
        self.norm = RunningFeatureNorm(cfg.n_features)
        self.register_buffer("fixed_policy", torch.tensor([0.0, _MERGE_BIAS, 0.0]))
        if cfg.use_controller:
            self.emb_proj = nn.Linear(cfg.emb_dim, cfg.emb_proj_dim)
            in_dim = cfg.n_features + len(DYNAMIC_INPUTS) + cfg.emb_proj_dim
            self.controller = AdaImprintController(AdaImprintConfig(
                input_dim=in_dim, hidden_dim=cfg.hidden_dim,
                max_log_importance_correction=cfg.max_log_correction,
                num_policy_actions=len(ACTIONS),
            ))
            for head in (self.controller.correction_head, self.controller.policy_head):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
            with torch.no_grad():
                self.controller.policy_head[-1].bias.copy_(self.fixed_policy)
            self.next_head = nn.Linear(cfg.hidden_dim + cfg.emb_dim, cfg.n_protos)
        else:
            self.emb_proj = self.controller = self.next_head = None

    @property
    def log_tau(self) -> float:
        return math.log(self.cfg.tau)

    @property
    def lam(self) -> Tensor:
        return torch.exp(self.log_lambda)

    @property
    def cons_threshold(self) -> Tensor:
        """Consolidation threshold, kept inside [cons_floor, 1]."""
        return self.cons_thr.clamp(self.cfg.cons_floor, 1.0)

    def merge_eligible(self, sim: float) -> bool:
        """Hard rule: the candidate counts iff sim >= threshold (as FIOOptimizer)."""
        return float(sim) >= float(self.cons_threshold.detach())

    def merge_eligibility(self, sim: float) -> Tensor:
        """Soft rule q = sigmoid((sim - threshold) / T), differentiable in the threshold."""
        return torch.sigmoid((torch.as_tensor(float(sim)) - self.cons_threshold) / self.cfg.cons_temp)

    def base_log_w(self, raw: Tensor) -> Tensor:
        """Parametric heuristic on raw (un-normalised) features [..., F]."""
        log1p_freq = raw[..., F_IDX["log1p_freq"]].clamp_min(1e-6)
        dt = raw[..., F_IDX["dt_sec"]].clamp_min(0.0)
        log1p_n = raw[..., F_IDX["log1p_n_protos"]]
        return (self.a_freq * torch.log(log1p_freq) - self.lam * dt
                - self.a_div * torch.log1p(log1p_n) + self.bias)

    def step(self, raw: Tensor, emb: Tensor, dynamic: Tensor,
             state: Optional[Tuple[Tensor, Tensor]] = None) -> Dict[str, Tensor]:
        """One timestep. raw [B,F], emb [B,D], dynamic [B, len(DYNAMIC_INPUTS)]."""
        log_w = self.base_log_w(raw)
        out: Dict[str, Tensor] = {}
        if self.controller is not None:
            x = torch.cat([self.norm(raw), dynamic, self.emb_proj(emb)], dim=-1)
            res = self.controller(x, state)
            log_w = log_w + res["log_importance_correction"]
            out["state"] = res["state"]
            out["h"] = res["state"][0]
            out["policy_logits"] = res["policy_logits"]
            out["correction"] = res["log_importance_correction"]
        else:
            out["policy_logits"] = self.fixed_policy.expand(raw.shape[0], -1)
        out["log_w"] = log_w
        out["gate"] = torch.sigmoid((log_w - self.log_tau) / self.cfg.gate_temp)
        return out

    def predict_next(self, h: Tensor, readout: Tensor) -> Tensor:
        return self.next_head(torch.cat([h, readout], dim=-1))

    def summary(self) -> Dict[str, float]:
        return {"a_freq": self.a_freq.item(), "a_div": self.a_div.item(),
                "lambda": self.lam.item(), "bias": self.bias.item(),
                "cons_thr": float(self.cons_threshold.detach())}


_WRITE_ONLY = (1.0, 0.0, 0.0)


def policy_probs(policy_logits: Tensor, has_candidate: bool,
                 eligibility: Optional[Tensor] = None) -> Tensor:
    """p(write, merge, update | retained). Write only when nothing to merge into.

    With a soft consolidation gate q (``eligibility``) the candidate is only
    usable with probability q:  P = (1 - q) * [1, 0, 0] + q * softmax(logits).
    """
    if not has_candidate:
        return torch.tensor(_WRITE_ONLY)
    pi = F.softmax(policy_logits, dim=-1)
    if eligibility is None:
        return pi
    return (1.0 - eligibility) * torch.tensor(_WRITE_ONLY) + eligibility * pi


def surprise_from_logits(next_logits: Optional[Tensor], proto: int, n_protos: int) -> float:
    if next_logits is None:
        return 0.0
    lp = F.log_softmax(next_logits.detach().reshape(-1), dim=-1)
    return float(-lp[proto] / math.log(n_protos))


def save_checkpoint(path: str, model: LearnableImportance, extra: Optional[Dict] = None) -> None:
    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict(),
                "extra": extra or {}}, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Tuple[LearnableImportance, Dict]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    model = LearnableImportance(LearnableImportanceConfig(**blob["config"]))
    missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
    # checkpoints from before the learnable consolidation gate lack cons_thr:
    # it stays at cons_thr_init (= the heuristic's 0.75)
    if unexpected or set(missing) - {"cons_thr"}:
        raise RuntimeError(f"{path}: missing {missing}, unexpected {unexpected}")
    model.eval()
    return model, blob.get("extra", {})
