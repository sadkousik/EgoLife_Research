#opti_fio_test.py
# FIO Optimizer — v6
#
# Importance scoring (revised methodology — interaction strength removed):
#
#            log(1 + ν) · exp(-λ ΔT)
#   w(f_t) = ────────────────────────
#                 1 + log(1 + N)
#
#   i.e. frequency × recency ÷ distinctiveness only. The α (interaction
#   strength) term no longer enters w(f_t). compute_interaction_strength()
#   is therefore no longer called by compute_weight().
#
# The Dynamic α table below is STILL maintained per FIO (process() Step 4)
# so that alpha_score stays available for downstream QA re-ranking — it is
# simply decoupled from the keep/drop importance decision.
#
#   α is learned dynamically from FIO structure alone (no labels needed):
#
#     Three structural proxy signals per action type:
#
#     1. FIELD RICHNESS SCORE
#        Does this action type tend to produce populated FIOs?
#        α_richness = mean(field_richness_score) across all FIOs of this type
#        where field_richness_score = weighted sum of non-null fields
#        (object=0.25, app=0.20, attributes=0.25, target_location=0.15, tool=0.10)
#
#     2. ATTRIBUTE ENTROPY
#        Does this action type carry consistent, informative attributes?
#        High entropy = diverse attribute values = genuinely informative
#        Low entropy = always empty or always same value = low signal
#        α_entropy = normalised entropy of attribute key-value distribution
#
#     3. ACTION-OBJECT STABILITY
#        Does this action type pair with a consistent object?
#        Stable pairing (set→timer, activate→timer) = intentional, high signal
#        Scattered pairing (look→phone/door/window/...) = background noise
#        α_stability = 1 - normalised entropy of object distribution
#
#   Final α for an action type:
#     α = w1·α_richness + w2·α_entropy + w3·α_stability
#     (weights: 0.4, 0.3, 0.3 — richness dominates, entropy + stability refine)
#
#   α is updated via EMA after every FIO so it adapts as more data arrives.
#   Cold start: α=0.5 until 3+ samples seen, then structural signals take over.
#
# All v5 fixes retained:
#   - is_redundant arg order fixed
#   - compute_weight receives fio
#   - clean attr serialization
#   - strength floor lowered to 0.1
#   - importance_score_old integration
#   - compression with best-field merging

import copy
from importlib.resources import path
import json
import math
import time
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer


# ================================================================
# Field helpers
# ================================================================

# Weights for field richness calculation (must sum to 1.0)
_FIELD_WEIGHTS = {
    "object":          0.25,
    "app":             0.20,
    "attributes":      0.25,
    "target_location": 0.15,
    "tool":            0.10,
    "source_location": 0.05,
}

def _parse_attrs(attrs) -> Dict:
    """
    Normalise the FIO 'attributes' field to a dict.

    Accepts:
      - dict (already parsed)              → returned as-is
      - JSON string '{"k": "v"}'            → parsed and returned
      - empty / "{}" / "null" / None        → empty dict
      - malformed                           → empty dict

    Use this before any code that expects attributes as a dict.
    Without it, the type check `isinstance(attrs, dict)` silently fails on
    every FIO produced by the build pipeline (which serialises attributes
    to a JSON string before storage).
    """
    if attrs is None or attrs == "" or attrs == "{}":
        return {}
    if isinstance(attrs, dict):
        return attrs
    if isinstance(attrs, str):
        try:
            parsed = json.loads(attrs)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError, TypeError):
            return {}
    return {}


def _flatten_attrs(attrs) -> str:
    attrs = _parse_attrs(attrs)
    if not attrs:
        return ""
    return " ".join(
        f"{k} {v}" for k, v in attrs.items()
        if v not in (None, "", [], {})
    )

def _field_richness_score(fio: Dict) -> float:
    """Weighted presence score across informative FIO fields. Range [0, 1]."""
    score = 0.0
    for f, w in _FIELD_WEIGHTS.items():
        val = fio.get(f)
        # Attributes need special handling: '{}' / '"{}"' / 'null' look populated
        # but are semantically empty. Parse before deciding.
        if f == "attributes":
            if _parse_attrs(val):
                score += w
        else:
            if val not in (None, "", {}, [], "null", "{}"):
                score += w
    return score

def _field_richness_val(val) -> float:
    """Heuristic richness of a single field value (for merge decisions)."""
    if val is None: return 0.0
    if isinstance(val, dict):
        return float(sum(1 for v in val.values() if v not in (None, "", [], {})))
    if isinstance(val, list): return float(len(val))
    if isinstance(val, str): return float(len(val.strip()))
    return 1.0

def _entropy(counts: Dict) -> float:
    """Normalised Shannon entropy of a frequency dict. Range [0, 1]."""
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0
    probs = [c / total for c in counts.values()]
    raw = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(counts))
    return raw / max_entropy if max_entropy > 0 else 0.0
# timestamp from FIO
def fio_to_seconds(fio):
    """
    Convert EgoLife timestamp to absolute seconds.

    Example:
      DAY1 + 11093015
      -> 11:09:30

    Last two digits are ignored.
    """

    day = int(str(fio["date"]).replace("DAY", ""))

    ts = str(fio["start_time"])[:-2]

    hh = int(ts[:2])
    mm = int(ts[2:4])
    ss = int(ts[4:6])

    return (
        (day - 1) * 24 * 3600 +
        hh * 3600 +
        mm * 60 +
        ss
    )
#end timestamp from FIO

def _action_snapshot(fio: Dict) -> Dict:
    """
    Minimal snapshot for action_sequence entries.
    Only what's needed to reconstruct the temporal narrative —
    who did what to which object, with what content, and when.
    Everything else lives in the canonical FIO.
    """
    return {
        "action":     fio.get("action"),
        "object":     fio.get("object"),
        "attributes": copy.deepcopy(fio.get("attributes") or {}),
        "start_time": fio.get("start_time"),
        "end_time":   fio.get("end_time"),
    }


# Identity fields define WHAT the interaction IS.
# These are NEVER overwritten on merge — the base FIO's identity is preserved.
# This is what prevents semantic Frankensteins like "walked forward + phone".
_IDENTITY_FIELDS = {"action", "object", "app", "persons"}

# Internal / score / housekeeping — never touched on merge
_MERGE_SKIP_FIELDS = {
    "id", "proto_id", "importance_score", "importance_score_old",
    "embedding", "_compressed_from", "_ctx_emb", "_action_emb", "_kept_at",
    "heuristic_score", "recency_score", "alpha_score", "optimized",
    "video_path", "date", "start_time", "end_time",
    "action_sequence",   # handled separately below
}


def _is_empty(val) -> bool:
    """Return True if a field value carries no information."""
    if val is None:
        return True
    if isinstance(val, str):
        s = val.strip().lower()
        return s in ("", "null", "none", "{}", "[]")
    if isinstance(val, (dict, list)):
        return len(val) == 0
    return False


def _merge_fios(base: Dict, incoming: Dict) -> Dict:
    """
    Merge an incoming FIO into a canonical (base) kept FIO.

    Design principle:
      The base FIO's identity (action, object, app, persons) is FROZEN.
      The base FIO was the first to be kept under this prototype — its
      fields define what this consolidated event "is". Subsequent absorbed
      FIOs only contribute their temporal snapshot to action_sequence,
      plus (optionally) fill missing peripheral fields the base was silent on.

    Why frozen identity:
      Per-field "richer wins" (the previous approach) treated FIOs as
      independent fields, which silently produced incoherent tuples like
      "walked forward + phone" when an incoming FIO with object=phone
      overwrote base.object="". Identity must remain a coherent unit.

    Field handling:
      - IDENTITY fields (action, object, app, persons)  → BASE WINS (always)
      - SKIP fields (id, scores, embeddings, timestamps) → BASE WINS (always)
      - attributes (dict)                                → DEEP MERGE
              (incoming fills missing keys only; base's values are preserved)
      - other peripheral fields (tool, locations, speech_content, ...)
              → BASE WINS unless base was empty, then incoming fills
      - action_sequence
              → APPEND incoming's snapshot to the running sequence
    """
    merged = copy.deepcopy(base)

    for key, i_val in incoming.items():
        # Identity is sacred — never overwrite
        if key in _IDENTITY_FIELDS:
            continue
        # Housekeeping fields — never touch
        if key in _MERGE_SKIP_FIELDS:
            continue

        b_val = merged.get(key)

        # attributes: deep-merge so incoming fills missing keys but
        # never overwrites a value the base already provided.
        if key == "attributes":
            b_attrs = _parse_attrs(b_val)
            i_attrs = _parse_attrs(i_val)
            for k, v in i_attrs.items():
                if k not in b_attrs and not _is_empty(v):
                    b_attrs[k] = v
            # Serialize back to JSON string to match storage convention
            merged[key] = json.dumps(b_attrs, ensure_ascii=False) if b_attrs else "{}"
            continue

        # Other peripheral fields: enrich-when-empty only
        if _is_empty(b_val) and not _is_empty(i_val):
            merged[key] = i_val

    # ── action_sequence: temporal narrative of the merge chain ──────────
    # First merge: seed with the canonical (base) snapshot, then append incoming.
    # Subsequent merges: just append incoming.
    existing_seq = base.get("action_sequence")
    if isinstance(existing_seq, str):
        try:
            existing_seq = json.loads(existing_seq)
        except (json.JSONDecodeError, TypeError):
            existing_seq = None
    if existing_seq:
        seq = list(existing_seq)
    else:
        seq = [_action_snapshot(base)]   # seed with canonical
    seq.append(_action_snapshot(incoming))
    merged["action_sequence"] = seq

    return merged


# ================================================================
# DynamicAlphaTable — semantic embedding version
#
# Key upgrade from string-keyed version:
#   Instead of looking up α by exact action string, we store
#   (action_embedding, α) pairs and interpolate at query time
#   using softmax-weighted cosine similarity.
#
#   Effect:
#     "look", "observe", "glance", "watch" → all semantically close
#     → share α influence → unseen synonyms get sensible prior
#     "set", "configure", "adjust"        → cluster together
#     "move", "walk", "step"              → cluster together
#
#   Lookup: get(query_emb) returns
#     Σ softmax(sim(query, centroid_i)) * α_i   over all clusters
#
#   Update: new FIO either merges into the nearest cluster
#     (if cosine_sim > CLUSTER_SIM_THRESHOLD) or spawns a new one.
#     Each cluster maintains:
#       centroid     — EMA-updated mean embedding of the cluster
#       alpha        — current α value for this cluster
#       label        — the most common action string (for display)
#       richness_ema — EMA of field richness for cluster members
#       attr_counts  — freq dict of "key=value" attr strings
#       obj_counts   — freq dict of object values seen
#       sample_count — total FIOs assigned to this cluster
#
# Three structural signals drive α (same as before):
#   1. field_richness  — how populated are FIOs in this cluster?
#   2. attr_entropy    — how informative/diverse are their attributes?
#   3. obj_stability   — how consistently does the cluster target one object?
# ================================================================

class DynamicAlphaTable:
    """
    Embedding-based dynamic α table.
    α is keyed by action embedding clusters, not exact strings.
    Semantically similar actions share and propagate α values.
    """

    # Signal weights (must sum to 1.0)
    W_RICHNESS  = 0.40
    W_ATTR_PRES = 0.30   # attr_presence: how often does this action carry attributes?
    W_STABILITY = 0.30

    # EMA smoothing for α and centroid updates
    EMA_ALPHA   = 0.2

    # Cosine similarity threshold to merge into existing cluster vs spawn new
    CLUSTER_SIM_THRESHOLD = 0.82

    # Softmax temperature for lookup interpolation
    # Lower = sharper (nearest cluster dominates)
    # Higher = smoother (all clusters contribute)
    SOFTMAX_TEMP = 0.1

    def __init__(self, cold_start_prior: float = 0.5):
        self.prior = cold_start_prior
        # MAX_CLUSTERS=None means unbounded.
        # Override via FIOOptimizer(max_alpha_clusters=N) if you want a hard cap.
        # The natural vocabulary of egocentric verbs bounds this anyway —
        # in 11K-FIO runs we observe ~50-150 distinct semantic clusters,
        # so no artificial cap is needed for correctness or memory.
        self.MAX_CLUSTERS: Optional[int] = None

        # List of clusters. Each cluster is a dict:
        # {
        #   "centroid":     np.ndarray,       EMA-updated mean embedding
        #   "alpha":        float,             current α for this cluster
        #   "label":        str,               most common action string
        #   "label_counts": Dict[str,int],     for picking best label
        #   "richness_ema": float,
        #   "attr_counts":  Dict[str,int],     "key=value" → count
        #   "obj_counts":   Dict[str,int],     object_str → count
        #   "sample_count": int,
        # }
        self.clusters: List[Dict] = []

    # ── internal cosine sim ───────────────────────────────────────
    @staticmethod
    def _cos(a, b) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    # ── find nearest cluster ──────────────────────────────────────
    def _nearest(self, emb) -> Tuple[int, float]:
        """Return (cluster_idx, cosine_sim) of nearest cluster, or (-1, -1)."""
        if not self.clusters:
            return -1, -1.0
        sims = [self._cos(c["centroid"], emb) for c in self.clusters]
        best = int(np.argmax(sims))
        return best, sims[best]

    # ── softmax over similarities ─────────────────────────────────
    def _softmax_weights(self, sims: List[float]) -> List[float]:
        arr = np.array(sims) / self.SOFTMAX_TEMP
        arr -= arr.max()   # numerical stability
        exp = np.exp(arr)
        return (exp / exp.sum()).tolist()

    # ── update: ingest one FIO ────────────────────────────────────
    def update(self, fio: Dict, action_emb) -> float:
        """
        Update α for the cluster nearest to action_emb.
        action_emb: pre-computed embedding of fio["action"].
        Returns the updated α for the matched/spawned cluster.
        """
        action  = (fio.get("action") or "unknown").lower().strip()
        richness = _field_richness_score(fio)

        best_idx, best_sim = self._nearest(action_emb)

        # When MAX_CLUSTERS is None (default), the cap never forces a merge.
        # When set to an int, exceeding that count forces a force-merge into
        # the nearest cluster — used only if you want a hard memory ceiling.
        at_cap = (
            self.MAX_CLUSTERS is not None
            and len(self.clusters) >= self.MAX_CLUSTERS
        )

        if best_sim >= self.CLUSTER_SIM_THRESHOLD or (at_cap and best_sim >= 0):
            # ── merge into existing cluster ───────────────────────
            c = self.clusters[best_idx]
            c["sample_count"] += 1

            # update centroid via EMA
            c["centroid"] = (
                (1 - self.EMA_ALPHA) * c["centroid"]
                + self.EMA_ALPHA * action_emb
            )
            # re-normalise centroid so it stays on the unit sphere
            norm = np.linalg.norm(c["centroid"])
            if norm > 1e-8:
                c["centroid"] = c["centroid"] / norm

            # update label
            c["label_counts"][action] = c["label_counts"].get(action, 0) + 1
            c["label"] = max(c["label_counts"], key=c["label_counts"].get)

            # update richness EMA
            c["richness_ema"] = (
                (1 - self.EMA_ALPHA) * c["richness_ema"]
                + self.EMA_ALPHA * richness
            )

            # update attr_presence EMA
            attrs_dict = _parse_attrs(fio.get("attributes"))
            has_attrs = 1.0 if any(
                v not in (None, "", [], {}) for v in attrs_dict.values()
            ) else 0.0
            c["attr_presence_ema"] = (
                (1 - self.EMA_ALPHA) * c["attr_presence_ema"]
                + self.EMA_ALPHA * has_attrs
            )

        else:
            # ── spawn new cluster ─────────────────────────────────
            attrs_dict = _parse_attrs(fio.get("attributes"))
            has_attrs = 1.0 if any(
                v not in (None, "", [], {}) for v in attrs_dict.values()
            ) else 0.0
            emb_norm = action_emb / (np.linalg.norm(action_emb) + 1e-8)
            c = {
                "centroid":          emb_norm.copy(),
                "alpha":             self.prior,
                "label":             action,
                "label_counts":      {action: 1},
                "richness_ema":      richness,
                "attr_presence_ema": has_attrs,   # 1.0 if first FIO had attrs
                "attr_counts":       {},
                "obj_counts":        {},
                "sample_count":      1,
            }
            self.clusters.append(c)
            best_idx = len(self.clusters) - 1

        # ── update attribute counts ───────────────────────────────
        attrs = _parse_attrs(fio.get("attributes"))
        if attrs:
            for k, v in attrs.items():
                if v not in (None, "", [], {}):
                    key = f"{k}={v}"
                    c["attr_counts"][key] = c["attr_counts"].get(key, 0) + 1
        else:
            c["attr_counts"]["__empty__"] = c["attr_counts"].get("__empty__", 0) + 1

        # ── update object counts ──────────────────────────────────
        obj_key = str(fio.get("object") or "__none__").lower().strip()
        c["obj_counts"][obj_key] = c["obj_counts"].get(obj_key, 0) + 1

        # ── compute structural signals ────────────────────────────
        # attr_presence: EMA of "did this action carry attributes?"
        # Works from sample 1 — set=1.0, look=0.0, no entropy needed.
        attr_presence = c["attr_presence_ema"]

        # obj_stability: how consistently does this cluster target a REAL
        # object? Object-free actions (walked, nodded, said) accumulate as
        # __none__ — they should score low, not high.
        #
        # We compute stability over real-object counts only, then scale by
        # the fraction of samples that actually had a real object. This
        # handles four cases cleanly:
        #   - all __none__       (locomotion/speech)   → 0.0
        #   - all one object     (set→timer)            → 1.0
        #   - all scattered objs (picked up→many)       → low (high entropy)
        #   - mix of __none__ + real                    → real_fraction · stability
        real_objs = {k: v for k, v in c["obj_counts"].items() if k != "__none__"}
        n_real    = sum(real_objs.values())
        n_total   = sum(c["obj_counts"].values())
        if n_real == 0 or n_total == 0:
            obj_stability = 0.0
        else:
            real_fraction = n_real / n_total
            real_entropy  = _entropy(real_objs)   # 0 if single object
            obj_stability = (1.0 - real_entropy) * real_fraction

        structural_alpha = (
            self.W_RICHNESS  * c["richness_ema"]
            + self.W_ATTR_PRES * attr_presence
            + self.W_STABILITY * obj_stability
        )

        # ── EMA update α ──────────────────────────────────────────
        c["alpha"] = max(0.05, min(1.0,
            (1 - self.EMA_ALPHA) * c["alpha"]
            + self.EMA_ALPHA * structural_alpha
        ))

        return c["alpha"]

    # ── get: interpolated α for a query embedding ─────────────────
    def get(self, action_emb) -> float:
        """
        Return α for a query action embedding.

        If only one cluster or all similarities are low:
            returns nearest cluster α (hard assignment).
        Otherwise:
            softmax-weighted interpolation over all cluster α values,
            so semantically nearby clusters share their α.

        Falls back to prior if no clusters exist yet.
        """
        if not self.clusters:
            return self.prior

        sims = [self._cos(c["centroid"], action_emb) for c in self.clusters]
        best_idx = int(np.argmax(sims))

        if len(self.clusters) == 1 or sims[best_idx] < 0.3:
            # hard assignment — only one meaningful match
            return self.clusters[best_idx]["alpha"]

        # softmax-weighted blend
        weights = self._softmax_weights(sims)
        alpha   = sum(w * c["alpha"] for w, c in zip(weights, self.clusters))
        return float(np.clip(alpha, 0.05, 1.0))

    # ── get_table: diagnostics ────────────────────────────────────
    def get_table(self) -> Dict[str, Dict]:
        """Return per-cluster diagnostics keyed by cluster label."""
        table = {}
        for i, c in enumerate(self.clusters):
            # Same obj_stability calculation as in update() — real-object-aware
            real_objs = {k: v for k, v in c["obj_counts"].items() if k != "__none__"}
            n_real    = sum(real_objs.values())
            n_total   = sum(c["obj_counts"].values())
            if n_real == 0 or n_total == 0:
                obj_stability = 0.0
            else:
                obj_stability = (1.0 - _entropy(real_objs)) * (n_real / n_total)

            # top_objects diagnostic: filter __none__ so the table shows
            # what the cluster actually manipulates. If a cluster is mostly
            # __none__, this list will be empty — which itself is informative.
            top = sorted(
                ((k, v) for k, v in c["obj_counts"].items() if k != "__none__"),
                key=lambda x: -x[1],
            )[:3]

            table[c["label"]] = {
                "cluster_id":       i,
                "alpha":            round(c["alpha"], 4),
                "samples":          c["sample_count"],
                "richness_ema":     round(c["richness_ema"], 4),
                "attr_presence":    round(c["attr_presence_ema"], 4),
                "obj_stability":    round(obj_stability, 4),
                "n_real_obj":       n_real,
                "n_total":          n_total,
                "label_counts":     c["label_counts"],
                "top_objects":      top,
            }
        return table



# ================================================================
# ACTION SYNONYM GROUPS
#
# Egocentric action verbs that are semantically interchangeable
# for the purposes of FIO redundancy detection.
# Two FIOs with the same object+app but synonym actions are treated
# as the same event and compressed together.
#
# Extend this list as new action verbs appear in your EgoLife data.
# Groups are derived from VerbNet / egocentric action ontologies.
# ================================================================
_ACTION_SYNONYM_GROUPS: List[set] = [
    # Visual perception
    {"look", "observe", "watch", "glance", "see", "notice", "view", "gaze", "stare", "peek"},
    # Manipulation — setting/configuring
    {"set", "configure", "adjust", "change", "modify", "assign", "select", "choose", "input", "enter"},
    # Activation / starting
    {"activate", "start", "launch", "begin", "trigger", "initiate", "turn_on", "power_on", "press", "tap"},
    # App opening
    {"open", "open_app", "launch_app"},
    # Holding / gripping
    {"hold", "grab", "pick_up", "take", "grasp", "grip"},
    # Placing / putting down
    {"put", "place", "put_down", "set_down", "drop", "lay"},
    # Locomotion
    {"move", "walk", "step", "go", "proceed", "navigate", "travel", "approach"},
    # Reading / checking
    {"read", "check", "review", "scan", "browse", "look_at"},
    # Drinking / eating
    {"drink", "sip", "consume", "eat", "taste"},
    # Talking / communicating
    {"talk", "speak", "say", "tell", "call", "phone"},
]

def _same_synonym_group(a1: str, a2: str) -> bool:
    """Return True if a1 and a2 belong to the same synonym group."""
    a1 = a1.lower().strip()
    a2 = a2.lower().strip()
    if a1 == a2:
        return True
    for group in _ACTION_SYNONYM_GROUPS:
        if a1 in group and a2 in group:
            return True
    return False


class FIOOptimizer:
    def __init__(
        self,
        delta_g_threshold: float = 0.75,
        tau_threshold: float = 0.3,
        delta_h_threshold: float = 0.5,#used in history
        lambda_decay: float = 0.01,
        redundancy_cooldown_sec: float = 1.0,
        compression_sim_threshold: float = 0.75,
        alpha_cold_start: float = 0.5,
        alpha_ema: float = 0.2,

        # ── Memory bounds ─────────────────────────────────────────
        # Sized for 7-day EgoLife dataset at ~200 FIOs/hr × 8hr/day
        # Total input: ~11,200 FIOs. After filtering ~30-50% kept.
        #
        # history_window       : sliding window for context encoding
        #                        3 is sufficient; no value in keeping more
        # max_prototypes       : cap on context-embedding prototype clusters.
        #                        After cap, always merge into nearest.
        # max_action_prototypes: cap on action embedding clusters.
        #                        None = unbounded (default). Natural language
        #                        vocabulary bounds this — typically saturates
        #                        at ~50-150 clusters even on 7-day runs.
        #                        Set to an int only for a hard memory ceiling.
        # max_object_interactions: per-object interaction history cap.
        #                        E[α] uses only the most recent N interactions
        #                        (recency bias — old interactions less relevant)
        # max_kept_for_compression: sliding window of kept FIOs checked
        #                        during compression targeting. Full kept_output
        #                        scan is O(n) and grows unbounded without this.
        # max_alpha_clusters   : cap on DynamicAlphaTable clusters.
        #                        None = unbounded (default). Same rationale
        #                        as max_action_prototypes — semantic action
        #                        vocabulary is naturally bounded.
        history_window: int                       = 5,
        max_prototypes: int                       = 200,
        max_action_prototypes: Optional[int]      = None,
        max_object_interactions: int              = 20,
        max_kept_for_compression: int             = 100,
        max_alpha_clusters: Optional[int]         = None,

        # ── Contextual embedding ────────────────────────────────────
        # context_history_weight (β): how much the recent-history buffer
        #   influences the contextual embedding for prototype clustering.
        #   ctx_emb = (1-β)·e_current + β·e_history, then unit-normalized.
        #   β = 0    — clustering uses current FIO content only
        #   β = 0.3  — current dominates (70%), history nudges (default)
        #   β = 0.5  — equal weight
        #   β = 1    — history dominates (recreates the original buggy
        #              concatenation behavior asymptotically)
        #
        # Why this exists: the original encode_with_context concatenated
        # history and current FIO text into one string before embedding.
        # On a 5-FIO history, the history dominated the string by a 4:1
        # ratio, which collapsed semantically distinct FIOs into the same
        # prototype cluster (Bug 4 — within-clip frequency inflation).
        # Separating the embeddings and combining with explicit weight
        # gives a tunable, position-invariant context representation.
        context_history_weight: float             = 0.3,

        # ── Diagnostics ─────────────────────────────────────────────
        # track_growth: when True, records pool sizes after each FIO
        #               into self._growth_log. Use save_growth_csv() to
        #               dump for plotting saturation curves. Off by
        #               default — adds ~250KB memory per 10K FIOs.
        track_growth: bool                        = False,

        # ── Stage-1 ablation (no_s1) ────────────────────────────────
        # importance_mode = "score"   → keep iff w(f_t) >= τ   (default; full model)
        # importance_mode = "uniform" → ABLATE prototype-based importance:
        #     keep each retained-eligible FIO with probability target_keep_rate
        #     (Bernoulli), ignoring w(f_t). Set target_keep_rate to the full
        #     model's measured total_kept / total_input so the AMOUNT kept
        #     matches and only the SELECTION is randomized. The stored
        #     importance_score is also neutralized so retrieval's w_imp term
        #     carries no prototype signal. uniform_seed makes the draw
        #     reproducible across the comparison.
        importance_mode: str                      = "score",
        target_keep_rate: float                   = 0.40,
        uniform_seed: int                         = 1234,
    ):
        # -------- Model --------
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # -------- Dynamic α table --------
        self.alpha_table = DynamicAlphaTable(cold_start_prior=alpha_cold_start)
        self.alpha_table.EMA_ALPHA   = alpha_ema
        self.alpha_table.MAX_CLUSTERS = max_alpha_clusters

        # -------- Prototype state — bounded --------
        self.prototypes:        List[Any]       = []   # capped at max_prototypes
        # ν_n — number of contextual embeddings absorbed into each prototype.
        # Indexed in lockstep with self.prototypes (never pruned here).
        # Used for the incremental centroid update in assign_prototype so that
        # g_n is the true running mean of its assigned embeddings, matching
        # eq:proto-centroid / eq:proto-update in the methodology. Kept separate
        # from proto_stats["count"] because that count is advanced in Step 5
        # (after assignment), whereas the centroid update needs the pre-update ν.
        self.proto_counts:      List[int]        = []   # ν_n, parallel to prototypes
        self.proto_stats:       Dict[int, Dict] = {}
        self.action_prototypes: List[Any]       = []   # capped at max_action_prototypes

        # -------- History — sliding window only --------
        # Never grows beyond history_window entries.
        # Use a deque for O(1) append/pop.
        from collections import deque
        self.history: deque = deque(maxlen=history_window)

        # -------- kept_output — full list (needed for dedup checks) --------
        # Embeddings (_ctx_emb, _action_emb) are stripped after the
        # compression window passes to reclaim memory.
        self.kept_output: List[Dict] = []

        # for t-SNE / UMAP visualization
        self.tsne_data = []
        # for horizontal bar for event_protype wrt to delta g for assiginig prototypes to events
        self.merge_count = 0          # TOTAL merges (genuine sim-driven + cap-forced)
        self.forced_merge_count = 0   # subset of merge_count forced by the prototype cap
        self.new_proto_count = 0

        # Sliding window index into kept_output for compression targeting.
        # Only the last max_kept_for_compression kept FIOs are scanned.
        self._compression_window = max_kept_for_compression

        # -------- Per-object interaction history — capped --------
        # Stores only the last max_object_interactions (action, emb) tuples.
        # E[α] uses recent interactions only — old ones are less informative.
        self._max_obj_interactions = max_object_interactions
        self.object_interactions: Dict[str, List[Tuple]] = defaultdict(list)

        # -------- Hyperparams --------
        self.DELTA_H                      = delta_h_threshold #used in history delta h calculation
        self.SIM_THRESHOLD                = delta_g_threshold #used in event prototype assignment delta g calculation
        self.KEEP_THRESHOLD               = tau_threshold # in importance score
        self.LAMBDA                       = lambda_decay
        self.REDUNDANCY_COOLDOWN_SEC      = redundancy_cooldown_sec
        self.COMPRESSION_SIM              = compression_sim_threshold
        self.SEMANTIC_REDUNDANCY_THRESHOLD = 0.60

        # β for contextual embedding combination — clipped to [0, 1].
        if not (0.0 <= context_history_weight <= 1.0):
            raise ValueError(
                f"context_history_weight must be in [0, 1], "
                f"got {context_history_weight}"
            )
        self.CONTEXT_HISTORY_WEIGHT = float(context_history_weight)

        # -------- Stage-1 ablation (uniform retention, no_s1) --------
        if importance_mode not in ("score", "uniform"):
            raise ValueError(
                f"importance_mode must be 'score' or 'uniform', got {importance_mode!r}"
            )
        self.IMPORTANCE_MODE  = importance_mode
        self.TARGET_KEEP_RATE = float(target_keep_rate)
        # Isolated RNG so the ablation draw is reproducible and does NOT
        # perturb any global numpy state used elsewhere.
        self._uniform_rng     = np.random.default_rng(uniform_seed)

        # -------- Memory bounds --------
        self._max_prototypes        = max_prototypes
        self._max_action_prototypes = max_action_prototypes

        # -------- Compression tracking --------
        self._compression_log: List[Dict] = []
        self._key_to_kept_idx: Dict[str, int] = {}

        # -------- Memory stats (for monitoring) --------
        self._stats = {
            "total_input":        0,
            "total_kept":         0,
            "total_compressed":   0,
            "total_dropped":      0,
            "proto_cap_hits":     0,   # times prototype cap was enforced
            "emb_strips":         0,   # times embeddings stripped from old kept FIOs
        }

        # -------- Growth tracking (opt-in) --------
        # When track_growth=True, each FIO appends a row to _growth_log.
        # Use save_growth_csv() at end of build to dump for plotting.
        self._track_growth: bool = track_growth
        self._growth_log: List[Dict] = []
        self.total_discovered_prototypes = 0 # for lifetime tracking (not reset when prototypes are pruned)
        self._fio_index: int = 0   # 1-indexed running counter
        # Header order — also used by save_growth_csv
        self._growth_columns: List[str] = [
            "fio_idx",
            "action_prototypes",
            "alpha_clusters",
            "context_prototypes",
            "object_keys",
            "kept_output",
        ]

    # ============================================================
    # Debug helper
    # ============================================================
    def debug(self, title: str, data=None):
        print("\n" + "-" * 20 + f" {title} " + "-" * 20)
        if data is not None:
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f"  {k}: {v}")
            else:
                print(" ", data)

    # ============================================================
    # Struct key
    # ============================================================
    @staticmethod
    def _struct_key(fio: Dict) -> str:
        action = (fio.get("action") or "").lower().strip()
        obj    = (fio.get("object") or "").lower().strip()
        app    = (fio.get("app")    or "").lower().strip()
        return f"{action}|{obj}|{app}"

    @staticmethod
    def _object_key(fio: Dict) -> str:
        return str(fio.get("object") or "__none__").lower().strip()

    @staticmethod
    def _event_label(fio: Dict) -> str:
        """
        Short human-readable name for an interaction, used for naming event
        prototypes in the t-SNE / UMAP visualizations. Prefers action+object,
        then action+app, then action+target_location, then bare action.
        """
        action = (fio.get("action") or "").strip()
        for field in ("object", "app", "target_location", "tool"):
            val = fio.get(field)
            if val not in (None, "", {}, [], "null", "{}"):
                return f"{action} {str(val).strip()}".strip()
        return action or "unknown"

    # ============================================================
    # Encoding
    # ============================================================
    def cosine_sim(self, a, b) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def encode_with_context_old_notUesdNow(self, fio: Dict) -> Any:
        """
        Compute the contextual embedding for an incoming FIO.

        Design (separate-embedding combination):

          ctx_emb = (1 - β) · e_current + β · e_history     normalized

        where:
          e_current : embedding of the current FIO's content fields
          e_history : embedding of the joined text of recent history FIOs
                      (empty when history is empty — then β term is skipped)
          β         : self.CONTEXT_HISTORY_WEIGHT, in [0, 1]

        Why this replaces the original concat-then-embed scheme:

          The original encode_with_context concatenated history_text and
          current_text into one string and embedded it as a whole. On a
          5-FIO history, the resulting string was ~80% history tokens.
          Sentence-encoder pooling treats tokens roughly democratically,
          so the embedding became history-dominated regardless of the
          current FIO's content. As a result, semantically distinct FIOs
          (e.g. "walked forward" arriving amid timer interactions) ended
          up clustered into the same context prototype, artificially
          inflating frequency counts in w(o) — Bug 4 from the trace.

          Separating the two embeddings and combining them with an
          explicit weight β makes the influence of history (a) bounded
          (b) tunable (c) ablatable (β=0 recovers a "current-only"
          baseline) (d) position-invariant w.r.t. how full history is.
        """
        # ── e_current: embed the current FIO's content ────────────────
        attr_str = _flatten_attrs(fio.get("attributes"))
        current_text = " ".join(filter(None, [
            fio.get("action") or "",
            fio.get("object") or "",
            fio.get("app") or "",
            fio.get("target_location") or "",
            attr_str,
        ])).strip()
        if not current_text:
            current_text = "unknown"   # degenerate FIO — embed something
        e_current = self.model.encode(current_text)

        # ── e_history: embed the joined history (only if history exists) ──
        history_text = " ".join([
            f"{h.get('action','')} {h.get('object','') or ''} {h.get('app','') or ''}"
            for h in list(self.history)
            if isinstance(h, dict)
        ]).strip()

        β = self.CONTEXT_HISTORY_WEIGHT

        if history_text and β > 0.0:
            e_history = self.model.encode(history_text)
            ctx_emb = (1.0 - β) * e_current + β * e_history
        else:
            # No history yet, or β = 0 — use current-only embedding.
            ctx_emb = e_current

        # ── Unit-normalize so cosine similarities are well-defined. ──
        # The weighted sum can have arbitrary magnitude; normalizing
        # ensures cosine_sim(a, b) is bounded in [-1, 1] regardless of β.
        norm = np.linalg.norm(ctx_emb)
        if norm > 1e-8:
            ctx_emb = ctx_emb / norm

        self.debug("Context Encoding", {
            "history_size": len(self.history),
            "history_text": history_text,
            "current_text": current_text,
            "β (history weight)": round(β, 3),
            "scheme": "weighted-combo" if (history_text and β > 0.0) else "current-only",
        })
        return ctx_emb

    #updated encoded with context for contextual embedding
    def encode_with_context(self, fio: Dict) -> Any:
        """
        Contextual embedding, per the methodology.
 
          e_t        = Encoder(a_t, o_t)                         (action + object)
          𝓗_t        = { e_j ∈ H_t : cos(e_t, e_j) > δ_h }       (related subset)
          Mean(𝓗_t)  = (1/|𝓗_t|) Σ_{e_j ∈ 𝓗_t} e_j
          ẽ_t        = (1 - β) · e_t + β · Mean(𝓗_t)             (normalized)
          ẽ_t        = e_t                          when 𝓗_t = ∅
 
        The history buffer H_t holds the raw event embeddings e_j of the k
        most recent interactions (stored as "_e_t" on each history entry).
        Context is built ONLY from the history items whose embedding is within
        δ_h of e_t — unrelated recent activity is filtered out, instead of the
        previous scheme that averaged ALL recent history in as joined text.
 
        e_t is stashed on fio["_e_t"] so the caller can enqueue it into the
        history buffer at the end of the loop (H_{t+1} = enqueue_k(H_t, e_t)).
        """
        # ── e_t = Encoder(a_t, o_t) ───────────────────────────────────
        a = (fio.get("action") or "").strip()
        o = (fio.get("object") or "").strip()
        event_text = " ".join(filter(None, [a, o])).strip() or "unknown"
        e_t = self.model.encode(event_text)
        fio["_e_t"] = e_t   # enqueued into H_{t+1} by the process() loop
 
        β       = self.CONTEXT_HISTORY_WEIGHT
        delta_h = self.DELTA_H
 
        # ── 𝓗_t : history embeddings within δ_h of e_t ────────────────
        related:   List[Any]   = []
        sims_dbg:  List[float] = []
        if β > 0.0:
            for h in self.history:
                if not isinstance(h, dict):
                    continue
                e_j = h.get("_e_t")
                if e_j is None:
                    continue
                s = self.cosine_sim(e_t, e_j)
                sims_dbg.append(round(s, 4))
                if s > delta_h:
                    related.append(e_j)
 
        # ── ẽ_t = (1-β)·e_t + β·Mean(𝓗_t) ; ẽ_t = e_t if 𝓗_t = ∅ ──────
        if related:
            mean_h  = np.mean(np.stack(related), axis=0)
            ctx_emb = (1.0 - β) * e_t + β * mean_h
            scheme  = f"filtered-mean (|𝓗_t|={len(related)})"
        else:
            ctx_emb = e_t
            scheme  = "current-only (𝓗_t = ∅)"
 
        # ── Unit-normalize so cosine similarities are well-defined. ──
        norm = np.linalg.norm(ctx_emb)
        if norm > 1e-8:
            ctx_emb = ctx_emb / norm
 
        self.debug("Context Encoding", {
            "history_size":      len(self.history),
            "event_text":        event_text,
            "δ_h":               round(delta_h, 3),
            "β (history weight)": round(β, 3),
            "sims_to_history":   sims_dbg,
            "n_related (>δ_h)":  len(related),
            "scheme":            scheme,
        })
        return ctx_emb

    def encode_action_only(self, fio: Dict) -> Any:
        action_text = fio.get("action") or ""
        if not action_text:
            print(f"  [WARN] Missing action: {fio.get('id', '?')}")
            return np.zeros(self.model.get_sentence_embedding_dimension())
        return self.model.encode(action_text)

    # ============================================================
    # Action normalization
    # ============================================================
    def normalize_action(self, fio: Dict) -> Any:
        emb = self.encode_action_only(fio)
        if not self.action_prototypes:
            self.debug("Action Prototype Init", "First action prototype created")
            self.action_prototypes.append(emb)
            return emb
        sims = [self.cosine_sim(p, emb) for p in self.action_prototypes]
        best_idx = int(np.argmax(sims))
        # at_cap is True only when an explicit int cap is set AND exceeded.
        # When self._max_action_prototypes is None (default), no force-merge.
        at_cap = (
            self._max_action_prototypes is not None
            and len(self.action_prototypes) >= self._max_action_prototypes
        )
        if sims[best_idx] > self.SIM_THRESHOLD or at_cap:
            self.action_prototypes[best_idx] = (
                (1 - self.alpha_table.EMA_ALPHA) * self.action_prototypes[best_idx]
                + self.alpha_table.EMA_ALPHA * emb
            )
            self.debug("Action Prototype Merged", {
                "prototype_id": best_idx, "similarity": round(sims[best_idx], 4)
            })
            return self.action_prototypes[best_idx]
        self.action_prototypes.append(emb)
        self.debug("Action Prototype New", {
            "prototype_id": len(self.action_prototypes) - 1
        })
        return emb

    # ============================================================
    # Prototype assignment — bounded
    # When cap is reached, always merge into nearest (no new spawning).
    # ============================================================
    def assign_prototype(self, emb) -> int:
        if not self.prototypes:
            self.new_proto_count += 1 # for horizontal bar visualization of proto assignment wrt delta g
            self.prototypes.append(emb)
            self.proto_counts.append(1)            # ν_1 = 1 (eq:proto-init)
            self.total_discovered_prototypes += 1
            self.debug("Prototype Init", "First prototype created")
            return 0
        sims = [self.cosine_sim(p, emb) for p in self.prototypes]
        best_idx = int(np.argmax(sims))
        self.debug("Prototype Similarity", {
            "similarities": [round(s, 4) for s in sims],
            "best": best_idx,
            "sim": round(sims[best_idx], 4),
        })
        if sims[best_idx] > self.SIM_THRESHOLD or len(self.prototypes) >= self._max_prototypes:
            if len(self.prototypes) >= self._max_prototypes and sims[best_idx] <= self.SIM_THRESHOLD:
                self._stats["proto_cap_hits"] += 1
                self.forced_merge_count += 1   # merge happened only because the cap was full
                self.debug("Prototype Cap", f"Cap {self._max_prototypes} reached — forced merge into {best_idx}")
            else:
                self.debug("Prototype Merge", f"Merged into prototype {best_idx}")
            self.merge_count += 1 # for horizontal bar visualization of proto assignment wrt delta g
            # Incremental centroid update (eq:proto-update):
            #   g_n ← (ν_n · g_n + ẽ_t) / (ν_n + 1),  then  ν_n ← ν_n + 1
            # ν_n is the count of embeddings already absorbed into prototype n,
            # so g_n stays the exact running mean of its members rather than the
            # recency-biased 50/50 EMA used previously.
            nu = self.proto_counts[best_idx]
            self.prototypes[best_idx] = (nu * self.prototypes[best_idx] + emb) / (nu + 1)
            self.proto_counts[best_idx] = nu + 1
            return best_idx

        self.new_proto_count += 1 # for horizontal bar visualization of proto assignment wrt delta g
        self.prototypes.append(emb)
        self.proto_counts.append(1)                # ν_{N+1} = 1 (eq:proto-init)
        self.total_discovered_prototypes += 1
        self.debug("Prototype New", f"New prototype: {len(self.prototypes) - 1}")
        return len(self.prototypes) - 1

    # ============================================================
    # Update stats
    # ============================================================
    def update_stats(self, proto_id: int, fio: Dict, emb=None):
        
        fio_ts=fio["_effective_ts"]#fio_to_seconds(fio) # fetch timestamp in seconds for logging

        if proto_id not in self.proto_stats:
            self.proto_stats[proto_id] = {
                "count": 0,
                "last_seen": fio_ts,#time.time(),
                "first_seen": fio_ts,#time.time(),
                "action_types": [],          # list of action strings (for E[α])
            }
        self.proto_stats[proto_id]["count"] += 1
        self.proto_stats[proto_id]["last_seen"] = fio_ts
        self.debug("Stats Update", {
            "proto_id": proto_id,
            "count": self.proto_stats[proto_id]["count"],
        })

    # ============================================================
    # E[α] — interaction strength per object (exact RQ5 formula)
    #
    # E_{i∈I(o)}[α_{type(i)}] = mean of α[emb_i] for each interaction
    # i in the history of object o.
    #
    # Uses embedding-based lookup so "look" and "observe" share α.
    # If object has no history yet, returns the cold-start prior.
    #
    # NOTE: No longer called by compute_weight() under the revised
    # methodology (α was removed from importance scoring). Retained for
    # diagnostics / optional re-ranking experiments only.
    # ============================================================
    def compute_interaction_strength(self, fio: Dict) -> Tuple[float, List[float]]:
        obj_key      = self._object_key(fio)
        interactions = self.object_interactions[obj_key]  # list of (action_str, emb)

        if not interactions:
            return self.alpha_table.prior, []

        # look up α for each past interaction using its embedding
        alphas = [self.alpha_table.get(emb) for _, emb in interactions]
        return float(np.mean(alphas)), alphas

    # ============================================================
    # Importance weight — revised methodology (no interaction strength)
    #
    #            log(1 + ν_{n(t)}) · exp(-λ ΔT)
    #   w(f_t) = ──────────────────────────────
    #                   1 + log(1 + N)
    #
    #   ν = recurrence count (frequency), ΔT = elapsed effective time
    #   since the prototype was last seen (recency), N = number of active
    #   event prototypes (distinctiveness).
    #
    #   The E_{i∈I(o)}[α_{type(i)}] interaction-strength term has been
    #   removed — importance is now frequency × recency ÷ distinctiveness
    #   only. The α table is still maintained (Step 4 in process()) so that
    #   alpha_score remains available for downstream QA re-ranking, but it
    #   no longer participates in the keep/drop importance decision.
    # ============================================================
    def compute_weight(self, proto_id: int, fio: Dict) -> Dict[str, float]:
        #stats    = self.proto_stats[proto_id]
        #f        = stats["count"]
        current_ts = fio["_effective_ts"]#fio_to_seconds(fio) #current timestamp in seconds from fio
        if proto_id not in self.proto_stats:
            f = 1
            delta_t = 0
            df = max(1, len(self.proto_stats))

        else:

            stats = self.proto_stats[proto_id]

            f = stats["count"]
            delta_t  = max(0, current_ts - stats["last_seen"])#time.time() - stats["last_seen"]
            df       = len(self.proto_stats)

        log_term       = math.log(1 + f)
        decay_term     = math.exp(-self.LAMBDA * delta_t)
        diversity_term = 1.0 / (1.0 + math.log(1 + df))

        # w(f_t) = log(1+ν) · exp(-λΔT) / (1 + log(1+N))
        # No interaction-strength (α) term — see method docstring above.
        total = log_term * decay_term * diversity_term

        breakdown = {
            "f (frequency)":      f,
            "log_term":           round(log_term, 4),
            "decay_term":         round(decay_term, 6),
            "df":                 df,
            "diversity_term":     round(diversity_term, 4),
            "TOTAL":              round(total, 4),
        }
        self.debug("Importance Score", breakdown)
        return {"score": total, "breakdown": breakdown}

    # ============================================================
    # Redundancy check — three-tier system
    #
    # Tier 1 (exact):    struct key match in kept_output
    #                    catches "look/phone" == "look/phone"
    #
    # Tier 2 (semantic): action_emb cosine sim against kept action_embs
    #                    catches "observe/phone" ≈ "look/phone"
    #                    only fires when object also matches
    #
    # Tier 3 (proto):    proto_id + same action + same attrs in history
    #                    legacy safety net
    # ============================================================
    def is_redundant(self, f: Dict, proto_id: int, action_emb=None) -> Optional[int]:
        """
        LEGACY — no longer used by the consolidation path.

        Superseded by _find_consolidation_target(), which searches the memory
        set M (kept_output) only. This method's Tier 3 scans self.history,
        which is NOT the memory set, so it does not match the methodology's
        "search the memory set M" definition of consolidation. Kept for
        reference / ablation; not called in the main process() loop.

        Check whether incoming FIO `f` is redundant with a kept FIO.

        Returns:
            int  — index into self.kept_output of the matched FIO (the one
                   `f` should be merged into), if redundancy was detected.
            None — `f` is unique across all three tiers.

        Truthiness preserved:
            `if self.is_redundant(...)` still works as before — None is falsy,
            any int (including 0) is truthy. But callers that want to know
            WHICH kept FIO matched can now read the integer directly.
            This is what lets compression target the correct FIO instead of
            re-deciding by ctx_emb similarity (which can pick the wrong one
            when contextual embeddings are dominated by recent history).
        """
        key = self._struct_key(f)
        obj = (f.get("object") or "").lower().strip()

        # ── Tier 1: exact struct key ───────────────────────────────
        for kept_idx, kept in enumerate(self.kept_output):
            if self._struct_key(kept) == key:
                self.debug("Redundancy Check", {
                    "tier": 1,
                    "reason": f"exact struct key '{key}' in kept_output",
                    "matched_kept_idx": kept_idx,
                    "redundant": True,
                })
                return kept_idx

        # ── Tier 2: synonym group + same object + same app ────────
        # Catches: "observe/phone" → merges into "look/phone"
        # Uses explicit synonym groups — zero threshold sensitivity.
        # Embedding sim kept as 2b fallback for out-of-vocabulary verbs.
        cur_app = (f.get("app") or "").lower().strip()
        cur_action = (f.get("action") or "").lower().strip()
        for kept_idx, kept in enumerate(self.kept_output):
            kept_obj = (kept.get("object") or "").lower().strip()
            kept_app = (kept.get("app") or "").lower().strip()
            if kept_obj != obj or kept_app != cur_app:
                continue   # different target — not the same event
            kept_action = (kept.get("action") or "").lower().strip()
            if _same_synonym_group(cur_action, kept_action):
                self.debug("Redundancy Check", {
                    "tier": "2a-synonym",
                    "reason": f"'{cur_action}' is synonym of '{kept_action}' "
                              f"with same obj='{obj}' app='{cur_app}'",
                    "matched_kept_idx": kept_idx,
                    "redundant": True,
                })
                return kept_idx
            # 2b: embedding fallback for OOV verbs not in synonym map
            kept_emb = kept.get("_action_emb")
            if action_emb is not None and kept_emb is not None:
                sim = self.cosine_sim(action_emb, kept_emb)
                if sim >= self.SEMANTIC_REDUNDANCY_THRESHOLD:
                    self.debug("Redundancy Check", {
                        "tier": "2b-embedding",
                        "reason": f"OOV verb sim={sim:.3f}>={self.SEMANTIC_REDUNDANCY_THRESHOLD} "
                                  f"with same obj='{obj}' app='{cur_app}'",
                        "matched_action": kept_action,
                        "matched_kept_idx": kept_idx,
                        "redundant": True,
                    })
                    return kept_idx

        # ── Tier 3: proto + action + attrs in recent history ───────
        # Note: tier 3 matches against HISTORY, not kept_output. The kept_idx
        # to merge into isn't directly known — fall back to compression-target
        # search by ctx_emb. This is acceptable because tier 3 already requires
        # very specific structural match (same proto, action, attributes).
        for h in list(self.history)[-5:]:
            if h.get("proto_id") == proto_id:
                if (h.get("action") == f.get("action") and
                        h.get("attributes") == f.get("attributes")):
                    self.debug("Redundancy Check", {
                        "tier": 3,
                        "reason": "same proto + same action + same attributes",
                        "matched_kept_idx": "(tier-3: search by ctx_emb)",
                        "redundant": True,
                    })
                    # Sentinel: -1 means "redundant, but search for target".
                    return -1

        self.debug("Redundancy Check", {
            "proto_id": proto_id, "reason": "unique — all tiers passed", "redundant": False
        })
        return None

    # ============================================================
    # Compression target
    # ============================================================
    def _find_compression_target(self, fio: Dict, ctx_emb) -> Optional[int]:
        """
        LEGACY — no longer used. _find_consolidation_target() now performs
        the memory scan with both equivalence conditions inline. Retained
        for reference only.

        Scan only the last _compression_window kept FIOs (not all of kept_output).
        This bounds the scan to O(compression_window) regardless of dataset size.
        """
        now = time.time()
        best_idx, best_sim = None, -1.0
        # only scan the tail of kept_output
        window_start = max(0, len(self.kept_output) - self._compression_window)
        for kept_idx in range(window_start, len(self.kept_output)):
            kept = self.kept_output[kept_idx]
            if (now - kept.get("_kept_at", 0.0)) > self.REDUNDANCY_COOLDOWN_SEC * 10:
                continue
            kept_emb = kept.get("_ctx_emb")
            if kept_emb is None:
                continue
            sim = self.cosine_sim(kept_emb, ctx_emb)
            if sim >= self.COMPRESSION_SIM and sim > best_sim:
                best_sim, best_idx = sim, kept_idx
        return best_idx

    # ============================================================
    # Consolidation target — methodology "Interaction Consolidation"
    #
    #   "Given an incoming FIO f_t, we search the memory set M for an
    #    existing memory entry describing the same interaction."
    #
    # M is the retained memory = self.kept_output. The history buffer is
    # NOT searched here — it only feeds the contextual embedding. We scan
    # memory entries and return the one equivalent to f_t, where equivalence
    # requires BOTH:
    #   (1) same semantic interaction pattern — exact struct key
    #       (action|object|app), OR a synonym-group action with the same
    #       object and app (OOV verbs fall back to action-embedding sim), AND
    #   (2) contextual embeddings exceed a similarity threshold
    #       cos(ctx_emb_f, ctx_emb_kept) >= COMPRESSION_SIM.
    #
    # Only called for RETAINED FIOs (w >= τ). Among all memory entries that
    # satisfy (1), returns the index of the one with the highest contextual
    # similarity that also clears (2). Returns None → no equivalent entry,
    # so the caller adds f_t as a new memory entry  M ← M ∪ {f_t}.
    #
    # Scan is bounded to the tail of M (the compression window): entries
    # older than that have had their embeddings stripped, so condition (2)
    # cannot be verified for them and they are skipped anyway.
    # ============================================================
    def _find_consolidation_target(
        self, fio: Dict, proto_id: int, action_emb, ctx_emb
    ) -> Optional[int]:
        cur_key    = self._struct_key(fio)
        cur_obj    = (fio.get("object") or "").lower().strip()
        cur_app    = (fio.get("app")    or "").lower().strip()
        cur_action = (fio.get("action") or "").lower().strip()

        best_idx, best_sim = None, -1.0
        window_start = max(0, len(self.kept_output) - self._compression_window)

        for idx in range(window_start, len(self.kept_output)):
            kept = self.kept_output[idx]

            # ── (1) same semantic interaction pattern? ──────────────
            same_pattern = False
            if self._struct_key(kept) == cur_key:
                same_pattern = True                       # exact action|object|app
            else:
                kept_obj = (kept.get("object") or "").lower().strip()
                kept_app = (kept.get("app")    or "").lower().strip()
                if kept_obj == cur_obj and kept_app == cur_app:
                    kept_action = (kept.get("action") or "").lower().strip()
                    if _same_synonym_group(cur_action, kept_action):
                        same_pattern = True               # synonym + same obj/app
                    else:
                        # OOV fallback: action-embedding similarity
                        kept_aemb = kept.get("_action_emb")
                        if action_emb is not None and kept_aemb is not None:
                            if self.cosine_sim(action_emb, kept_aemb) >= self.SEMANTIC_REDUNDANCY_THRESHOLD:
                                same_pattern = True
            if not same_pattern:
                continue

            # ── (2) contextual-embedding similarity gate ────────────
            kept_ctx = kept.get("_ctx_emb")
            if kept_ctx is None:
                continue   # embedding stripped (too old) → cannot verify → skip
            sim = self.cosine_sim(ctx_emb, kept_ctx)
            if sim >= self.COMPRESSION_SIM and sim > best_sim:
                best_sim, best_idx = sim, idx

        if best_idx is not None:
            self.debug("Consolidation Match", {
                "matched_memory_idx": best_idx,
                "ctx_sim":            round(best_sim, 4),
            })
        else:
            self.debug("Consolidation Match", {
                "reason": "no equivalent memory entry — new entry",
            })
        return best_idx

    # ============================================================
    # Main processing loop
    # ============================================================
    def process(self, fios: List[Dict]) -> Dict:
        """
        Returns:
          {
            "kept":                list of retained (possibly merged) FIOs,
            "compressed_sequence": list of compression events,
            "alpha_table":         final α table with per-type diagnostics
          }
        """
        for idx, f in enumerate(fios):
            print("\n" + "=" * 60)
            print(f"🔹 Processing FIO {idx + 1}")
            # Pseudo timestamp:
            # same clip timestamp + FIO order offset

            f["_effective_ts"] = (
                    fio_to_seconds(f)
                    + self._fio_index
            )
            # print("RAW FIO:", {
            #     k: v for k, v in f.items()
            #     if k not in ("video_path", "embedding", "_ctx_emb")
            # })

            # ── Step 1: normalize action ───────────────────────────
            action_emb = self.normalize_action(f)
            print(f"\n[Step 1] action={f.get('action')} | emb[:3]={action_emb[:3].round(4)}")

            # ── Step 2: context encoding ───────────────────────────
            ctx_emb = self.encode_with_context(f)
            f["_ctx_emb"] = ctx_emb
            print(f"[Step 2] ctx_emb[:3]={ctx_emb[:3].round(4)}")

            # ── Step 3: prototype assignment ───────────────────────
            proto_id = self.assign_prototype(ctx_emb)
            f["proto_id"] = proto_id
            print(f"\n[Step 3] proto_id={proto_id} | total_protos={len(self.prototypes)}")

            #storing data for t-SNE / UMAP visualization (optional)
            self.tsne_data.append({
                "proto_id": proto_id,
                "embedding": ctx_emb.tolist(),
                "name": self._event_label(f),   # human-readable event name
            })
            # ── Step 3.5 removed ───────────────────────────────────
            # Redundancy is no longer checked here. Per the methodology,
            # consolidation runs ONLY on retained FIOs (w >= τ), so the
            # equivalence test now lives in Step 7 below.

            # ── Step 4: update stats + α table + object interactions ──
            #self.update_stats(proto_id, ctx_emb)
            self._stats["total_input"] += 1

            # Register interaction — capped per object (recency bias)
            # Only last max_object_interactions kept per object
            obj_key  = self._object_key(f)
            action   = (f.get("action") or "unknown").lower().strip()
            obj_list = self.object_interactions[obj_key]
            obj_list.append((action, action_emb))
            if len(obj_list) > self._max_obj_interactions:
                obj_list.pop(0)  # drop oldest — recency bias

            # Update α for this cluster from the FIO's structure
            # Pass action_emb so table uses semantic clusters, not string keys
            new_alpha = self.alpha_table.update(f, action_emb)
            print(f"[Step 4] α[{action}] updated → {new_alpha:.4f}")

            # ── Growth tracking (opt-in via track_growth=True) ────────
            # Capture pool sizes AFTER all state updates for this FIO.
            # The saturation curve is best read after Step 4 because:
            #   - action_prototypes was updated in Step 1
            #   - context prototypes (self.prototypes) was updated in Step 3
            #   - alpha_table.clusters was updated in Step 4
            #   - object_interactions was updated in Step 4
            # kept_output is captured pre-Step-6 — i.e. before this FIO is
            # either kept/merged/dropped. That's intentional: we want the
            # growth curve to reflect the cumulative cluster vocabulary,
            # not the keep/drop decisions which are downstream of clustering.
            self._fio_index += 1
            if self._track_growth:
                self._growth_log.append({
                    "fio_idx":            self._fio_index,
                    "action_prototypes":  len(self.action_prototypes),
                    "alpha_clusters":     len(self.alpha_table.clusters),
                    "context_prototypes": len(self.prototypes),
                    "object_keys":        len(self.object_interactions),
                    "kept_output":        len(self.kept_output),
                })

            # ── Step 5: Importance scoring ─────────────────────────
            #   w(f_t) = log(1+ν)·exp(-λΔT) / (1+log(1+N))
            # Stats are READ by compute_weight, then ADVANCED by update_stats.
            # NOTE: under the uniform (no_s1) ablation, w(f_t) is still computed
            # (for logging + consolidation bookkeeping) but is NOT used for the
            # keep/drop decision.
            weight_result = self.compute_weight(proto_id, f)
            self.update_stats(proto_id, f, ctx_emb)
            score     = weight_result["score"]

            # ── Stage-1 ablation toggle ────────────────────────────
            if self.IMPORTANCE_MODE == "uniform":
                # Ablate prototype-based importance: keep with fixed probability,
                # ignoring w(f_t). Selection randomized; amount matches full model
                # when TARGET_KEEP_RATE = full-model total_kept/total_input.
                keep_flag = bool(self._uniform_rng.random() < self.TARGET_KEEP_RATE)
                print(f"\n[Step 5] UNIFORM | keep_rate={self.TARGET_KEEP_RATE} "
                      f"| retained={keep_flag} | (score={score:.4f} unused)")
            else:
                keep_flag = score >= self.KEEP_THRESHOLD   # retained when w >= τ
                print(f"\n[Step 5] score={score:.4f} | threshold={self.KEEP_THRESHOLD} | retained={keep_flag}")

            # ── Step 6: Threshold filter ───────────────────────────
            # Records with w(f_t) < τ are discarded from long-term memory.
            # They are NOT eligible for consolidation — they contribute
            # nothing to any memory entry.
            if not keep_flag:
                self._stats["total_dropped"] += 1
                print(f"❌ Discarded (w={score:.4f} < τ={self.KEEP_THRESHOLD}) "
                      f"— below threshold, no consolidation")

            else:
                # ── Step 7: Interaction consolidation ──────────────
                # Search the memory set M (kept_output) — NOT the history
                # buffer — for an equivalent entry (same semantic pattern
                # AND ctx-emb similarity > threshold).
                #   - found  → consolidate f into it (no new entry).
                #   - none   → create a new memory entry  M ← M ∪ {f}.
                target_idx = self._find_consolidation_target(
                    f, proto_id, action_emb, ctx_emb
                )

                if target_idx is not None:
                    # ── Consolidate into existing memory entry ─────
                    raw_snap = {k: v for k, v in f.items()
                                if k not in ("_ctx_emb", "embedding", "_action_emb")}
                    self.kept_output[target_idx] = _merge_fios(
                        self.kept_output[target_idx], f
                    )
                    self.kept_output[target_idx]["_kept_at"] = time.time()
                    self._compression_log[target_idx]["collapsed"].append({
                        "raw_fio": raw_snap,
                        "reason":  "consolidated",
                        "score":   round(score, 4),
                    })
                    self._stats["total_compressed"] += 1
                    print(f"🔀 CONSOLIDATED into memory[{target_idx}] "
                          f"(retained, w={score:.4f})")

                else:
                    # ── New memory entry  M ← M ∪ {f} ──────────────
                    # Under uniform (no_s1) ablation, neutralize the stored
                    # importance to a constant so retrieval's w_imp term carries
                    # no prototype-derived signal. A constant leaves ranking
                    # order unaffected by importance — i.e. importance fully
                    # ablated, not just the keep decision.
                    f["importance_score"] = (
                        1.0 if self.IMPORTANCE_MODE == "uniform" else score
                    )
                    f["_kept_at"]         = time.time()
                    f["_action_emb"]      = action_emb   # for tier-2 equivalence
                    f["_ctx_emb"]         = ctx_emb       # for ctx-emb equivalence gate
                    # action_sequence only added when something consolidates in

                    kept_idx = len(self.kept_output)
                    self.kept_output.append(f)
                    self._compression_log.append({"kept_fio_index": kept_idx, "collapsed": []})
                    self._key_to_kept_idx[self._struct_key(f)] = kept_idx
                    self._stats["total_kept"] += 1

                    # ── Strip embeddings outside compression window ──
                    # Embeddings are large (384-dim float32 = 1.5KB each).
                    # Once a kept FIO is beyond the compression window it can
                    # never be a consolidation target again, so embeddings
                    # are safe to drop.
                    strip_idx = len(self.kept_output) - 1 - self._compression_window
                    if strip_idx >= 0:
                        old = self.kept_output[strip_idx]
                        if "_ctx_emb" in old or "_action_emb" in old:
                            old.pop("_ctx_emb", None)
                            old.pop("_action_emb", None)
                            self._stats["emb_strips"] += 1

                    print(f"\n[Step 7] ✅ NEW MEMORY ENTRY | score={score:.4f}")
                    print(f"[SUMMARY] FIO {idx+1} | proto={proto_id} | score={score:.3f} "
                          f"| retained=True | new_entry=True "
                          f"| α[{action}]={new_alpha:.3f} "
                          f"| mem: protos={len(self.prototypes)} "
                          f"kept={len(self.kept_output)} "
                          f"obj_keys={len(self.object_interactions)}")

            # ── Update history buffer  H_{t+1} = enqueue_k(H_t, e_t) ──
            # Done for EVERY record, AFTER the consolidation decision
            # (methodology: history is "subsequently updated"). Placing it
            # here is what prevents f from matching itself in the tier-3
            # history scan inside _find_consolidation_target above.
            self.history.append({**f, "embedding": ctx_emb})

        # ── Clean output ───────────────────────────────────────────
        INTERNAL = {"_ctx_emb", "_kept_at", "_action_emb", "embedding"}
        clean_kept = []
        for i, fio in enumerate(self.kept_output):
            clean = {k: v for k, v in fio.items() if k not in INTERNAL}
            # drop action_sequence if nothing was merged in (no narrative to tell)
            if not self._compression_log[i]["collapsed"]:
                clean.pop("action_sequence", None)
            clean_kept.append(clean)
        compressed_sequence = []
        for i, entry in enumerate(self._compression_log):
            if entry["collapsed"]:
                entry["canonical"] = clean_kept[i]
                compressed_sequence.append(entry)

        alpha_table = self.alpha_table.get_table()

        # ── Final summary ──────────────────────────────────────────
        print("\n" + "=" * 60)
        print(f"FINAL: {len(clean_kept)} kept | "
              f"{sum(len(e['collapsed']) for e in compressed_sequence)} compressed | "
              f"from {len(fios)} input FIOs")
        # Render None caps as "∞" so the diagnostic stays human-readable
        action_cap = self._max_action_prototypes if self._max_action_prototypes is not None else "∞"
        alpha_cap  = self.alpha_table.MAX_CLUSTERS if self.alpha_table.MAX_CLUSTERS is not None else "∞"
        print(f"MEMORY: protos={len(self.prototypes)}/{self._max_prototypes} | "
              f"action_protos={len(self.action_prototypes)}/{action_cap} | "
              f"obj_keys={len(self.object_interactions)} | "
              f"alpha_clusters={len(self.alpha_table.clusters)}/{alpha_cap} | "
              f"proto_cap_hits={self._stats['proto_cap_hits']} | "
              f"emb_strips={self._stats['emb_strips']}")
        print("=" * 60)
        # for i, fio in enumerate(clean_kept):
        #     n_merged = len(self._compression_log[i]["collapsed"])
        #     tag = f" [+{n_merged} merged]" if n_merged else ""
        #     print(f"  [{i}]{tag} {fio.get('action')} / "
        #           f"obj={fio.get('object')} / app={fio.get('app')} / "
        #           f"attrs={fio.get('attributes')} | score={fio.get('importance_score',0):.3f}")

        # print("\n📊 Dynamic α table (final):")
        # print(f"  {'action':<18} {'α':>6}  {'samples':>7}  {'richness':>9}  "
        #       f"{'attr_pres':>9}  {'obj_stab':>9}  synonyms / top_objects")
        # for action, info in sorted(alpha_table.items(), key=lambda x: -x[1]["alpha"]):
        #     syns = [k for k in info["label_counts"] if k != action]
        #     syn_str = f"[+{syns}]" if syns else ""
        #     print(f"  {action:<18} {info['alpha']:>6.3f}  {info['samples']:>7}  "
        #           f"{info['richness_ema']:>9.3f}  {info['attr_presence']:>9.3f}  "
        #           f"{info['obj_stability']:>9.3f}  {syn_str or info['top_objects']}")

        return {
            "kept":                clean_kept,
            "compressed_sequence": compressed_sequence,
            "alpha_table":         alpha_table,
        }

    # for horizonntal bar visualization of proto assignment wrt delta g
    def export_merge_statistics(
        self,
        file_name="merge_stats.json"
    ):
        import json
        import os

        total = self.merge_count + self.new_proto_count

        if total == 0:
            return

        # merge_count is the TOTAL of merges; split it into genuine (sim-driven)
        # and cap-forced so the δ_g→merge story isn't polluted by cap pressure.
        forced_merge_count  = self.forced_merge_count
        genuine_merge_count = self.merge_count - forced_merge_count

        merge_rate         = 100 * self.merge_count        / total   # total (kept for compat)
        genuine_merge_rate = 100 * genuine_merge_count     / total
        forced_merge_rate  = 100 * forced_merge_count      / total
        new_rate           = 100 * self.new_proto_count    / total
        # share of merges that were cap-forced (NaN-safe)
        forced_share = (100 * forced_merge_count / self.merge_count) if self.merge_count else 0.0

        row = {
            "delta_g": self.SIM_THRESHOLD,
            "merge_rate": round(merge_rate, 2),                  # total merges (genuine + forced)
            "genuine_merge_rate": round(genuine_merge_rate, 2),  # sim-driven merges only ← use this for δ_g
            "forced_merge_rate": round(forced_merge_rate, 2),    # merges caused by the cap
            "new_rate": round(new_rate, 2),
            "merge_count": self.merge_count,
            "genuine_merge_count": genuine_merge_count,
            "forced_merge_count": forced_merge_count,
            "forced_share_of_merges": round(forced_share, 2),    # % of merges that were cap-forced
            "new_proto_count": self.new_proto_count,
            "num_prototypes": len(self.prototypes),
            "max_prototypes": self._max_prototypes,              # so you can see if the cap could bind
        }

        path = f"./t_sne/{file_name}"
        os.makedirs(os.path.dirname(path), exist_ok=True)   # self-create ./t_sne/ (+ any subdir)
        if os.path.exists(path):

            with open(path) as f:
                data = json.load(f)

        else:
            data = []

        # replace existing row for same delta_g
        data = [
            x for x in data
            if x["delta_g"] != self.SIM_THRESHOLD
        ]

        data.append(row)

        data.sort(
            key=lambda x: x["delta_g"]
        )

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(
            f"[INFO] Merge statistics saved -> {path}"
        )


    # store tsne data to EgoRAG/t_sne
    def export_tsne_data(self, file_name="prototype_tsne_data.json"):
        import json
        from collections import Counter

        # Representative name per prototype = most common member event label.
        # Aligned by index with `prototypes`. Plot scripts use this as the
        # default event name (still overridable manually at plot time).
        proto_name_votes = defaultdict(Counter)
        for a in self.tsne_data:
            proto_name_votes[a["proto_id"]][a.get("name", "")] += 1

        prototype_names = [
            (proto_name_votes[i].most_common(1)[0][0]
             if proto_name_votes.get(i) else f"g{i}")
            for i in range(len(self.prototypes))
        ]

        payload = {
            "assignments": self.tsne_data,
            "prototypes": [
                p.tolist() for p in self.prototypes
            ],
            "prototype_names": prototype_names,
        }

        import os
        path = f"./t_sne/{file_name}"
        os.makedirs(os.path.dirname(path), exist_ok=True)   # self-create ./t_sne/ (+ any subdir)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"[INFO] t-SNE data saved -> {path}")


    # ============================================================
    # Growth tracking — save CSV for saturation-curve plotting
    # ============================================================
    def save_growth_csv(self, path: str) -> None:
        """
        Dump the recorded growth log to a CSV file.

        Columns: fio_idx, action_prototypes, alpha_clusters,
                 context_prototypes, object_keys, kept_output.

        No-op if track_growth was not enabled at construction time
        (warns to stdout in that case so silent failures don't happen).

        Use plot_growth_curve.py to read this CSV and produce the
        saturation plot — the empirical defense of unbounding the
        action/alpha cluster caps.
        """
        if not self._track_growth:
            print(f"[growth] track_growth=False; nothing to save. "
                  f"Pass track_growth=True to FIOOptimizer to enable.")
            return
        if not self._growth_log:
            print(f"[growth] log is empty; nothing to save.")
            return

        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._growth_columns)
            writer.writeheader()
            writer.writerows(self._growth_log)
        print(f"[growth] saved {len(self._growth_log)} rows → {path}")


# ================================================================
# Sample FIOs — used as fallback when real data file is not found
# ================================================================
SAMPLE_FIOS = [
    {
        "id": "DAY1-001", "date": "DAY1", "person": "Jake",
        "action": "look", "object": "phone",
        "location": None, "source_location": None, "target_location": None,
        "tool": None, "app": None, "attributes": {}, "modality": "visual",
        "importance_score_old": 0.65, "importance_bucket": "medium",
    },
    {
        "id": "DAY1-002", "date": "DAY1", "person": "Jake",
        "action": "open_app", "object": "phone",
        "location": None, "source_location": None, "target_location": None,
        "tool": None, "app": "timer", "attributes": {}, "modality": "visual",
        "importance_score_old": 0.85, "importance_bucket": "high",
    },
    {
        "id": "DAY1-003", "date": "DAY1", "person": "Jake",
        "action": "set", "object": "timer",
        "location": None, "source_location": None, "target_location": None,
        "tool": None, "app": None,
        "attributes": {"duration": "10 seconds"}, "modality": "visual",
        "importance_score_old": 0.75, "importance_bucket": "high",
    },
    {
        "id": "DAY1-004", "date": "DAY1", "person": "Jake",
        "action": "activate", "object": "timer",
        "location": None, "source_location": None, "target_location": None,
        "tool": None, "app": None, "attributes": {}, "modality": "visual",
        "importance_score_old": 0.75, "importance_bucket": "high",
    },
    {
        "id": "DAY1-005", "date": "DAY1", "person": "Jake",
        "action": "move", "object": None,
        "location": None, "source_location": None, "target_location": "forward",
        "tool": None, "app": None, "attributes": {}, "modality": "visual",
        "importance_score_old": 0.75, "importance_bucket": "high",
    },
    {
        "id": "DAY1-006", "date": "DAY1", "person": "Jake",
        "action": "observe", "object": "phone",
        "location": None, "source_location": None, "target_location": None,
        "tool": None, "app": None, "attributes": {}, "modality": "visual",
        "importance_score_old": 0.65, "importance_bucket": "medium",
    },
]


# ================================================================
# Entry point
# ================================================================
if __name__ == "__main__":
    import json
    import os

    # ── Load FIOs ─────────────────────────────────────────────────
    REAL_DATA_PATH = "/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/recovered_fio_08_04_2026.json"

    if os.path.exists(REAL_DATA_PATH):
        print(f"[INFO] Loading real FIO data from: {REAL_DATA_PATH}")
        with open(REAL_DATA_PATH, "r") as f:
            fios = json.load(f)
        print(f"[INFO] Loaded {len(fios)} FIOs")
    else:
        print(f"[WARN] Real data not found at: {REAL_DATA_PATH}")
        print(f"[INFO] Falling back to {len(SAMPLE_FIOS)} sample FIOs")
        fios = SAMPLE_FIOS

    # ── Run optimizer ──────────────────────────────────────────────
    optimizer = FIOOptimizer(
        delta_h_threshold=0.75, # used in history similarity for consolidation
        delta_g_threshold=0.75, # used in proto assignment
        tau_threshold=0.2,#used in score
        lambda_decay=0.01,
        redundancy_cooldown_sec=1.0,
        compression_sim_threshold=0.75,
        alpha_cold_start=0.5,
        alpha_ema=0.2,
        # Memory bounds — tuned for 7-day EgoLife (8hr/day, ~200 FIOs/hr)
        history_window=5,
        max_prototypes=200,
        max_action_prototypes=None,       # unbounded — natural vocabulary
        max_object_interactions=20,
        max_kept_for_compression=100,
        max_alpha_clusters=None,          # unbounded — natural vocabulary
    )

    result = optimizer.process(fios)
    optimizer.export_tsne_data(
        "prototype_tsne_data.json"
    )

    # ── Build full output payload ──────────────────────────────────
    n_input      = len(fios)
    n_kept       = len(result["kept"])
    n_compressed = sum(len(e["collapsed"]) for e in result["compressed_sequence"])
    n_dropped    = n_input - n_kept - n_compressed

    # NOTE: compressed_sequence and alpha_table are included now for analysis.
    # Remove them from output_payload when no longer needed.
    output_payload = {
        # ── Core output ───────────────────────────────────────────
        "kept": result["kept"],

        # ── TEMPORARY: remove when analysis is done ───────────────
        "compressed_sequence": result["compressed_sequence"],   # TODO: remove
        "alpha_table":         result["alpha_table"],           # TODO: remove
        "stats": {                                              # TODO: remove
            "total_input":        n_input,
            "total_kept":         n_kept,
            "total_compressed":   n_compressed,
            "total_dropped":      n_dropped,
            "compression_rate":   round(1 - n_kept / max(n_input, 1), 4),
            "kept_rate":          round(n_kept / max(n_input, 1), 4),
            "memory": {
                "prototypes":       len(optimizer.prototypes),
                "action_prototypes": len(optimizer.action_prototypes),
                "alpha_clusters":   len(optimizer.alpha_table.clusters),
                "object_keys":      len(optimizer.object_interactions),
                "proto_cap_hits":   optimizer._stats["proto_cap_hits"],
                "emb_strips":       optimizer._stats["emb_strips"],
            },
        },
    }

    # ── Save output ────────────────────────────────────────────────
    out_path = os.path.join(
        os.path.dirname(REAL_DATA_PATH) if os.path.exists(REAL_DATA_PATH) else ".",
        "optimized_fios.json"
    )
    with open(out_path, "w") as f:
        json.dump(output_payload, f, indent=2, default=str)
    print(f"\n[INFO] Output saved → {out_path}")

    # ── Print summary to stdout ────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("STATS SUMMARY")
    print("=" * 60)
    print(json.dumps(output_payload["stats"], indent=2, default=str))

    print("\n" + "=" * 60)
    print("ALPHA TABLE")   # TODO: remove block when done
    print("=" * 60)
    print(json.dumps(output_payload["alpha_table"], indent=2, default=str))

    print("\n" + "=" * 60)
    print(f"COMPRESSED SEQUENCE ({len(result['compressed_sequence'])} events)")   # TODO: remove block
    print("=" * 60)
    print(json.dumps(output_payload["compressed_sequence"], indent=2, default=str))

    print("\n" + "=" * 60)
    print(f"KEPT FIOs ({n_kept})")
    print("=" * 60)
    print(json.dumps(output_payload["kept"], indent=2, default=str))