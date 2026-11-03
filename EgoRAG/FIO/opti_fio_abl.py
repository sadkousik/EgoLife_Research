#opti_fio_test.py
# FIO Optimizer — v6
#
# New in v6: Dynamic α table implementing the exact RQ5 formula
#
#   w(o) = log(1+f(o)) · exp(-λΔt) · 1/(1+log(1+df(o))) · E_{i∈I(o)}[α_{type(i)}]
#
#   The last term E_{i∈I(o)}[α_{type(i)}] is now computed correctly:
#     - For each interaction i in object o's history, look up α[action_type]
#     - Average them → interaction strength for that object
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

def _flatten_attrs(attrs) -> str:
    if not attrs or not isinstance(attrs, dict):
        return ""
    return " ".join(
        f"{k} {v}" for k, v in attrs.items()
        if v not in (None, "", [], {})
    )

def _field_richness_score(fio: Dict) -> float:
    """Weighted presence score across informative FIO fields. Range [0, 1]."""
    return sum(
        w for f, w in _FIELD_WEIGHTS.items()
        if fio.get(f) not in (None, "", {}, [])
    )

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


def _merge_fios(base: Dict, incoming: Dict) -> Dict:
    SKIP = {
        "proto_id", "importance_score", "importance_score_old",
        "embedding", "_compressed_from", "_ctx_emb", "_kept_at",
        "action_sequence",   # handled separately below
    }
    merged = copy.deepcopy(base)
    for key, i_val in incoming.items():
        if key in SKIP:
            continue
        b_val = base.get(key)
        if isinstance(b_val, dict) or isinstance(i_val, dict):
            merged[key] = {**(b_val if isinstance(b_val, dict) else {}),
                           **(i_val if isinstance(i_val, dict) else {})}
        elif _field_richness_val(i_val) > _field_richness_val(b_val):
            merged[key] = i_val

    # Accumulate action_sequence.
    # First merge: seed with canonical snapshot + incoming.
    # Subsequent merges: just append incoming.
    existing_seq = base.get("action_sequence")
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
        self.MAX_CLUSTERS = 50  # overridden by FIOOptimizer.__init__

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

        if best_sim >= self.CLUSTER_SIM_THRESHOLD or (
            len(self.clusters) >= self.MAX_CLUSTERS and best_sim >= 0
        ):
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
            has_attrs = 1.0 if (
                fio.get("attributes") and isinstance(fio.get("attributes"), dict)
                and any(v not in (None, "", [], {}) for v in fio["attributes"].values())
            ) else 0.0
            c["attr_presence_ema"] = (
                (1 - self.EMA_ALPHA) * c["attr_presence_ema"]
                + self.EMA_ALPHA * has_attrs
            )

        else:
            # ── spawn new cluster ─────────────────────────────────
            has_attrs = 1.0 if (
                fio.get("attributes") and isinstance(fio.get("attributes"), dict)
                and any(v not in (None, "", [], {}) for v in fio["attributes"].values())
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
        attrs = fio.get("attributes")
        if attrs and isinstance(attrs, dict):
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
        obj_entropy   = _entropy(c["obj_counts"])
        obj_stability = 1.0 - obj_entropy

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
            obj_entropy   = _entropy(c["obj_counts"])
            table[c["label"]] = {
                "cluster_id":       i,
                "alpha":            round(c["alpha"], 4),
                "samples":          c["sample_count"],
                "richness_ema":     round(c["richness_ema"], 4),
                "attr_presence":    round(c["attr_presence_ema"], 4),  # replaces attr_H
                "obj_stability":    round(1.0 - obj_entropy, 4),
                "label_counts":     c["label_counts"],
                "top_objects":      sorted(
                    c["obj_counts"].items(), key=lambda x: -x[1]
                )[:3],
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
        sim_threshold: float = 0.75,
        keep_threshold: float = 0.3,
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
        # max_prototypes       : cap on embedding prototype clusters.
        #                        After cap, always merge into nearest.
        # max_action_prototypes: cap on action embedding clusters
        # max_object_interactions: per-object interaction history cap.
        #                        E[α] uses only the most recent N interactions
        #                        (recency bias — old interactions less relevant)
        # max_kept_for_compression: sliding window of kept FIOs checked
        #                        during compression targeting. Full kept_output
        #                        scan is O(n) and grows unbounded without this.
        # max_alpha_clusters   : cap on DynamicAlphaTable clusters
        history_window: int            = 5,
        max_prototypes: int            = 200,
        max_action_prototypes: int     = 50,
        max_object_interactions: int   = 20,
        max_kept_for_compression: int  = 100,
        max_alpha_clusters: int        = 50,

        # ── Component ablation flags (Study B) ─────────────────────
        # Setting any of these to True sets that term to the
        # multiplicative identity (1.0), effectively removing it
        # from the weight formula w(f_i).
        # Note: "disable_recency" is equivalent to lambda_decay=0,
        # provided for explicit control.
        disable_frequency: bool        = False,
        disable_recency: bool          = False,
        disable_distinctiveness: bool  = False,
        disable_interaction: bool      = False,
    ):
        # -------- Model --------
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # -------- Dynamic α table --------
        self.alpha_table = DynamicAlphaTable(cold_start_prior=alpha_cold_start)
        self.alpha_table.EMA_ALPHA   = alpha_ema
        self.alpha_table.MAX_CLUSTERS = max_alpha_clusters

        # -------- Prototype state — bounded --------
        self.prototypes:        List[Any]       = []   # capped at max_prototypes
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

        # Sliding window index into kept_output for compression targeting.
        # Only the last max_kept_for_compression kept FIOs are scanned.
        self._compression_window = max_kept_for_compression

        # -------- Per-object interaction history — capped --------
        # Stores only the last max_object_interactions (action, emb) tuples.
        # E[α] uses recent interactions only — old ones are less informative.
        self._max_obj_interactions = max_object_interactions
        self.object_interactions: Dict[str, List[Tuple]] = defaultdict(list)

        # -------- Hyperparams --------
        self.SIM_THRESHOLD                = sim_threshold
        self.KEEP_THRESHOLD               = keep_threshold
        self.LAMBDA                       = lambda_decay
        self.REDUNDANCY_COOLDOWN_SEC      = redundancy_cooldown_sec
        self.COMPRESSION_SIM              = compression_sim_threshold
        self.SEMANTIC_REDUNDANCY_THRESHOLD = 0.60

        # -------- Component ablation flags (Study B) --------
        self.DISABLE_FREQUENCY       = disable_frequency
        self.DISABLE_RECENCY         = disable_recency
        self.DISABLE_DISTINCTIVENESS = disable_distinctiveness
        self.DISABLE_INTERACTION     = disable_interaction
        if any([disable_frequency, disable_recency,
                disable_distinctiveness, disable_interaction]):
            disabled = []
            if disable_frequency: disabled.append("FREQUENCY")
            if disable_recency: disabled.append("RECENCY")
            if disable_distinctiveness: disabled.append("DISTINCTIVENESS")
            if disable_interaction: disabled.append("INTERACTION")
            print(f"[FIOOptimizer ABLATION] Disabled components: {', '.join(disabled)}")

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

    # ============================================================
    # Encoding
    # ============================================================
    def cosine_sim(self, a, b) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def encode_with_context(self, fio: Dict) -> Any:
        context_text = " ".join([
            f"{h.get('action','')} {h.get('object','') or ''} {h.get('app','') or ''}"
            for h in list(self.history)   # deque → list for safe iteration/slicing
            if isinstance(h, dict)
        ])
        attr_str = _flatten_attrs(fio.get("attributes"))
        current_text = " ".join(filter(None, [
            fio.get("action") or "",
            fio.get("object") or "",
            fio.get("app") or "",
            fio.get("target_location") or "",
            attr_str,
        ])).strip()
        full_text = (context_text + " [SEP] " + current_text).strip()
        self.debug("Context Encoding", {
            "history_size": len(self.history),
            "context_text": context_text,
            "current_text": current_text,
        })
        return self.model.encode(full_text)

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
        if sims[best_idx] > self.SIM_THRESHOLD or len(self.action_prototypes) >= self._max_action_prototypes:
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
            self.prototypes.append(emb)
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
                self.debug("Prototype Cap", f"Cap {self._max_prototypes} reached — forced merge into {best_idx}")
            else:
                self.debug("Prototype Merge", f"Merged into prototype {best_idx}")
            self.prototypes[best_idx] = (self.prototypes[best_idx] + emb) / 2
            return best_idx
        self.prototypes.append(emb)
        self.debug("Prototype New", f"New prototype: {len(self.prototypes) - 1}")
        return len(self.prototypes) - 1

    # ============================================================
    # Update stats
    # ============================================================
    def update_stats(self, proto_id: int, emb=None):
        if proto_id not in self.proto_stats:
            self.proto_stats[proto_id] = {
                "count": 0,
                "last_seen": time.time(),
                "first_seen": time.time(),
                "action_types": [],          # list of action strings (for E[α])
            }
        self.proto_stats[proto_id]["count"] += 1
        self.proto_stats[proto_id]["last_seen"] = time.time()
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
    # Importance weight — exact RQ5 formula
    #
    # w(o) = log(1+f(o)) · exp(-λΔt) · 1/(1+log(1+df(o))) · E[α]
    # ============================================================
    def compute_weight(self, proto_id: int, fio: Dict) -> Dict[str, float]:
        stats    = self.proto_stats[proto_id]
        f        = stats["count"]
        delta_t  = time.time() - stats["last_seen"]
        df       = len(self.proto_stats)

        log_term       = math.log(1 + f)
        decay_term     = math.exp(-self.LAMBDA * delta_t)
        diversity_term = 1.0 / (1.0 + math.log(1 + df))

        # E_{i∈I(o)}[α_{type(i)}] — the RQ5 interaction strength term
        e_alpha, alpha_breakdown = self.compute_interaction_strength(fio)
        e_alpha = max(0.05, e_alpha)   # floor — never fully silence an object

        # ── Component ablation (Study B): set disabled terms to identity 1.0 ──
        if self.DISABLE_FREQUENCY:
            log_term = 1.0
        if self.DISABLE_RECENCY:
            decay_term = 1.0
        if self.DISABLE_DISTINCTIVENESS:
            diversity_term = 1.0
        if self.DISABLE_INTERACTION:
            e_alpha = 1.0

        total = log_term * decay_term * diversity_term * e_alpha

        breakdown = {
            "f (frequency)":      f,
            "log_term":           round(log_term, 4),
            "decay_term":         round(decay_term, 6),
            "df":                 df,
            "diversity_term":     round(diversity_term, 4),
            "E[α] (interaction)": round(e_alpha, 4),
            "α breakdown":        [round(a, 4) for a in alpha_breakdown],
            "TOTAL":              round(total, 4),
        }
        self.debug("Importance Score (RQ5)", breakdown)
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
    def is_redundant(self, f: Dict, proto_id: int, action_emb=None) -> bool:
        key = self._struct_key(f)
        obj = (f.get("object") or "").lower().strip()

        # ── Tier 1: exact struct key ───────────────────────────────
        for kept in self.kept_output:
            if self._struct_key(kept) == key:
                self.debug("Redundancy Check", {
                    "tier": 1,
                    "reason": f"exact struct key '{key}' in kept_output",
                    "redundant": True,
                })
                return True

        # ── Tier 2: synonym group + same object + same app ────────
        # Catches: "observe/phone" → merges into "look/phone"
        # Uses explicit synonym groups — zero threshold sensitivity.
        # Embedding sim kept as 2b fallback for out-of-vocabulary verbs.
        cur_app = (f.get("app") or "").lower().strip()
        cur_action = (f.get("action") or "").lower().strip()
        for kept in self.kept_output:
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
                    "redundant": True,
                })
                return True
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
                        "redundant": True,
                    })
                    return True

        # ── Tier 3: proto + action + attrs in recent history ───────
        for h in list(self.history)[-5:]:
            if h.get("proto_id") == proto_id:
                if (h.get("action") == f.get("action") and
                        h.get("attributes") == f.get("attributes")):
                    self.debug("Redundancy Check", {
                        "tier": 3,
                        "reason": "same proto + same action + same attributes",
                        "redundant": True,
                    })
                    return True

        self.debug("Redundancy Check", {
            "proto_id": proto_id, "reason": "unique — all tiers passed", "redundant": False
        })
        return False

    # ============================================================
    # Compression target
    # ============================================================
    def _find_compression_target(self, fio: Dict, ctx_emb) -> Optional[int]:
        """
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
            print("RAW FIO:", {
                k: v for k, v in f.items()
                if k not in ("video_path", "embedding", "_ctx_emb")
            })

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

            # ── Step 3.5: redundancy BEFORE stats update ───────────
            redundant_flag = self.is_redundant(f, proto_id, action_emb)
            print(f"[Step 3.5] Redundant: {redundant_flag}")

            # ── Step 4: update stats + α table + object interactions ──
            self.update_stats(proto_id, ctx_emb)
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

            # ── Step 5: compute w(o) using RQ5 formula ─────────────
            weight_result = self.compute_weight(proto_id, f)
            score     = weight_result["score"]
            keep_flag = score > self.KEEP_THRESHOLD

            print(f"\n[Step 5] score={score:.4f} | threshold={self.KEEP_THRESHOLD} | keep={keep_flag}")

            # ── Always update history — deque auto-drops oldest ────
            self.history.append({**f, "embedding": ctx_emb})

            # ── Step 6: filtering + compression ────────────────────
            if not keep_flag or redundant_flag:
                reason = "low_importance" if not keep_flag else "redundant"
                target_idx = self._find_compression_target(f, ctx_emb)
                if target_idx is not None:
                    raw_snap = {k: v for k, v in f.items()
                                if k not in ("_ctx_emb", "embedding", "_action_emb")}
                    self.kept_output[target_idx] = _merge_fios(
                        self.kept_output[target_idx], f
                    )
                    self.kept_output[target_idx]["_kept_at"] = time.time()
                    self._compression_log[target_idx]["collapsed"].append({
                        "raw_fio": raw_snap,
                        "reason":  reason,
                        "score":   round(score, 4),
                    })
                    self._stats["total_compressed"] += 1
                    print(f"❌ {reason} → 🔀 COMPRESSED into kept[{target_idx}]")
                else:
                    self._stats["total_dropped"] += 1
                    print(f"❌ Dropped ({reason}) — no merge target")
                continue

            # ── Step 7: keep ───────────────────────────────────────
            f["importance_score"] = score
            f["_kept_at"]         = time.time()
            f["_action_emb"]      = action_emb   # for tier-2 redundancy
            f["_ctx_emb"]         = ctx_emb       # for compression targeting
            # action_sequence only added when something merges in (_merge_fios)

            kept_idx = len(self.kept_output)
            self.kept_output.append(f)
            self._compression_log.append({"kept_fio_index": kept_idx, "collapsed": []})
            self._key_to_kept_idx[self._struct_key(f)] = kept_idx
            self._stats["total_kept"] += 1

            # ── Strip embeddings from kept FIOs outside compression window ──
            # Embeddings are large (384-dim float32 = 1.5KB each).
            # Once a kept FIO is beyond the compression window it can never
            # be a compression target again, so embeddings are safe to drop.
            strip_idx = len(self.kept_output) - 1 - self._compression_window
            if strip_idx >= 0:
                old = self.kept_output[strip_idx]
                if "_ctx_emb" in old or "_action_emb" in old:
                    old.pop("_ctx_emb", None)
                    old.pop("_action_emb", None)
                    self._stats["emb_strips"] += 1

            print(f"\n[Step 7] ✅ KEPT | score={score:.4f}")
            print(f"[SUMMARY] FIO {idx+1} | proto={proto_id} | score={score:.3f} "
                  f"| keep={keep_flag} | redundant={redundant_flag} "
                  f"| α[{action}]={new_alpha:.3f} "
                  f"| mem: protos={len(self.prototypes)} "
                  f"kept={len(self.kept_output)} "
                  f"obj_keys={len(self.object_interactions)}")

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
        print(f"MEMORY: protos={len(self.prototypes)}/{self._max_prototypes} | "
              f"action_protos={len(self.action_prototypes)}/{self._max_action_prototypes} | "
              f"obj_keys={len(self.object_interactions)} | "
              f"alpha_clusters={len(self.alpha_table.clusters)}/{self.alpha_table.MAX_CLUSTERS} | "
              f"proto_cap_hits={self._stats['proto_cap_hits']} | "
              f"emb_strips={self._stats['emb_strips']}")
        print("=" * 60)
        for i, fio in enumerate(clean_kept):
            n_merged = len(self._compression_log[i]["collapsed"])
            tag = f" [+{n_merged} merged]" if n_merged else ""
            print(f"  [{i}]{tag} {fio.get('action')} / "
                  f"obj={fio.get('object')} / app={fio.get('app')} / "
                  f"attrs={fio.get('attributes')} | score={fio.get('importance_score',0):.3f}")

        print("\n📊 Dynamic α table (final):")
        print(f"  {'action':<18} {'α':>6}  {'samples':>7}  {'richness':>9}  "
              f"{'attr_pres':>9}  {'obj_stab':>9}  synonyms / top_objects")
        for action, info in sorted(alpha_table.items(), key=lambda x: -x[1]["alpha"]):
            syns = [k for k in info["label_counts"] if k != action]
            syn_str = f"[+{syns}]" if syns else ""
            print(f"  {action:<18} {info['alpha']:>6.3f}  {info['samples']:>7}  "
                  f"{info['richness_ema']:>9.3f}  {info['attr_presence']:>9.3f}  "
                  f"{info['obj_stability']:>9.3f}  {syn_str or info['top_objects']}")

        return {
            "kept":                clean_kept,
            "compressed_sequence": compressed_sequence,
            "alpha_table":         alpha_table,
        }


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
        sim_threshold=0.75,
        keep_threshold=0.2,
        lambda_decay=0.01,
        redundancy_cooldown_sec=1.0,
        compression_sim_threshold=0.75,
        alpha_cold_start=0.5,
        alpha_ema=0.2,
        # Memory bounds — tuned for 7-day EgoLife (8hr/day, ~200 FIOs/hr)
        history_window=5,
        max_prototypes=200,
        max_action_prototypes=50,
        max_object_interactions=20,
        max_kept_for_compression=100,
        max_alpha_clusters=50,
    )

    result = optimizer.process(fios)

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