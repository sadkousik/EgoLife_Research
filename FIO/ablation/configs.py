"""Ablation and sweep configurations for ``scripts/build_memory.py``.

Every row starts from ``BASE_OPT_KWARGS`` (the locked full-model config) and
overrides only the factor under study. A sweep then overrides one more knob.

    ABLATION=no_s3 python scripts/build_memory.py
    SWEEP_PARAM=tau_threshold SWEEP_VAL=0.3 python scripts/build_memory.py
    ABLATION=no_recency SWEEP_PARAM=max_prototypes SWEEP_VAL=400 python scripts/build_memory.py

Only edit ``BASE_OPT_KWARGS`` if you also re-run ``full``: every other row's
number is compared against it.
"""

import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Dict

# ── Locked full-model configuration ──────────────────────────────────────────
BASE_OPT_KWARGS: Dict[str, Any] = dict(
    delta_g_threshold          = 0.75,      # δ_g prototype-assignment similarity
    tau_threshold              = 0.2,       # τ importance keep threshold
    delta_h_threshold          = 0.3,       # δ_h history similarity filter (Stage-1 ctx)
    lambda_decay               = 1.146e-6,  # λ recency decay
    redundancy_cooldown_sec    = 1.0,
    context_history_weight     = 0.5,       # β history weight in ẽ_t
    compression_sim_threshold  = 0.75,      # Stage-3 consolidation ctx-sim gate
    action_ema                 = 0.2,       # EMA rate for a capped action prototype
    history_window             = 5,
    max_prototypes             = 3000,      # N_max (ALICE 7-day replay: 2,581 uncapped)
    max_action_prototypes      = 50,        # cap on action-embedding prototypes
    max_kept_for_compression   = 100,
    track_growth               = True,
    keep_first_sighting        = True,      # novelty rule: always keep a new prototype's 1st sighting
)

# ── Ablation rows ────────────────────────────────────────────────────────────
# "opt" overrides BASE_OPT_KWARGS. Each row changes exactly one factor.
ABLATIONS: Dict[str, Dict[str, Any]] = {
    "full": {
        "desc": "All stages on (baseline anchor).",
        "opt":  {},
    },

    # Stage ablations ---------------------------------------------------------
    "no_ctx": {
        # Stage-1 sub-component: contextual encoding. β=0 → ẽ_t = e_t, so the
        # history buffer / δ_h no longer influence clustering.
        "desc": "Contextual encoding off (β = 0).",
        "opt":  {"context_history_weight": 0.0},
    },
    "no_s1": {
        # Stage 1: prototype-based importance → UNIFORM retention at the SAME
        # keep-rate as the full model. Set target_keep_rate to the full model's
        # measured total_kept / total_input (0.4 is a placeholder).
        "desc": "Prototype importance → uniform retention at matched keep-rate.",
        "opt":  {"importance_mode": "uniform", "target_keep_rate": 0.4},
    },
    "no_s2": {
        # Stage 2: importance pruning. w(f_t) > 0 always, so τ = -1 keeps every
        # FIO. w(f_t) is still computed, so retrieval's w_imp term has values.
        "desc": "Importance pruning off (τ → keep-all; w_imp still computed).",
        "opt":  {"tau_threshold": -1.0},
    },
    "no_s3": {
        # Stage 3: consolidation. Cosine sim maxes at 1.0, so a 2.0 gate means
        # no FIO is ever merged — every retained FIO is its own memory entry.
        "desc": "Interaction consolidation off (never merge).",
        "opt":  {"compression_sim_threshold": 2.0},
    },
    "s3_only": {
        # S1 off + S2 off (uniform draw at p=1.0 keeps everything, stored
        # importance neutralised), S3 on. Isolates what consolidation alone
        # does to a fully unfiltered store. Keep-rate is 1.0 by design.
        "desc": "Only consolidation active (S1 off, S2 off, keep-all + merge).",
        "opt":  {"importance_mode": "uniform", "target_keep_rate": 1.0,
                 "compression_sim_threshold": 0.75},
    },

    # Term ablations of w(f_t) = log(1+ν) · exp(-λΔT) / (1 + log(1+N)) --------
    # The disabled factor becomes 1.0; τ is unchanged, so the keep-rate moves
    # with the score scale — compare kept counts in build_report.json.
    "no_freq": {
        "desc": "Frequency term off (log(1+ν) → 1).",
        "opt":  {"disable_frequency": True},
    },
    "no_recency": {
        "desc": "Recency term off (exp(-λΔT) → 1; same as λ = 0).",
        "opt":  {"disable_recency": True},
    },
    "no_distinct": {
        "desc": "Distinctiveness term off (1/(1+log(1+N)) → 1).",
        "opt":  {"disable_distinctiveness": True},
    },
}

# Knobs that only the sweep should touch (documented for the launcher).
SWEEPABLE = sorted(BASE_OPT_KWARGS)


@dataclass
class AblationConfig:
    name: str
    desc: str
    run_tag: str
    opt_kwargs: Dict[str, Any] = field(default_factory=dict)
    sweep_param: str = ""
    sweep_value: Any = None


def _cast(param: str, raw: str) -> Any:
    """Cast a sweep value string to the type of the base default."""
    t = type(BASE_OPT_KWARGS[param])
    if t is bool:
        return raw.lower() in ("1", "true", "yes")
    try:
        return t(raw)
    except ValueError:
        raise ValueError(f"could not cast SWEEP_VAL={raw!r} to {t.__name__} for {param}") from None


def resolve(ablation: str = "full", sweep_param: str = "", sweep_val: str = "",
            run_tag: str = "") -> AblationConfig:
    """Build the optimizer kwargs for one run: base → ablation row → sweep."""
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ABLATION {ablation!r}; choose one of {list(ABLATIONS)}")
    row = ABLATIONS[ablation]
    opt = {**BASE_OPT_KWARGS, **row["opt"]}

    value = None
    if sweep_param:
        if sweep_param not in BASE_OPT_KWARGS:
            raise ValueError(f"SWEEP_PARAM {sweep_param!r} is not a base knob; choose one of {SWEEPABLE}")
        if sweep_val == "":
            raise ValueError(f"SWEEP_PARAM={sweep_param} is set but SWEEP_VAL is empty")
        value = _cast(sweep_param, sweep_val)
        opt[sweep_param] = value

    tag = run_tag or (ablation + (f"__{sweep_param}={sweep_val}" if sweep_param else ""))
    return AblationConfig(ablation, row["desc"], tag, opt, sweep_param, value)


def from_env() -> AblationConfig:
    """Read ABLATION, SWEEP_PARAM, SWEEP_VAL and RUN_TAG from the environment."""
    return resolve(
        ablation=os.environ.get("ABLATION", "full"),
        sweep_param=os.environ.get("SWEEP_PARAM", ""),
        sweep_val=os.environ.get("SWEEP_VAL", ""),
        run_tag=os.environ.get("RUN_TAG", ""),
    )


def check_supported(optimizer_cls, opt_kwargs: Dict[str, Any]) -> None:
    """Fail early if a row passes a kwarg the optimizer does not accept."""
    accepted = set(inspect.signature(optimizer_cls.__init__).parameters)
    unknown = [k for k in opt_kwargs if k not in accepted]
    if unknown:
        raise ValueError(f"{optimizer_cls.__name__} does not accept {unknown}")


if __name__ == "__main__":
    # python ablation/configs.py  → list every row and what it changes
    for name, row in ABLATIONS.items():
        print(f"{name:12} {row['desc']}")
        for k, v in row["opt"].items():
            print(f"{'':12}   {k} = {v!r}  (base {BASE_OPT_KWARGS.get(k, '—')!r})")
