# fio/schema.py
# ─────────────────────────────────────────────────────────────────────────────
# Canonical FIO field definitions, validation, document-string builder,
# and recency score computation.
#
# This file has NO side effects on import. It defines:
#   FIO_FIELDS          — ordered list of all field names with metadata
#   CHROMADB_FIELDS     — subset that goes into ChromaDB metadata
#   build_document_string(fio) → str   for semantic embedding
#   validate_fio(fio)          → dict  cleaned & type-safe for ChromaDB
#   compute_recency_score(fio, t_max_hours) → float
#   compute_heuristic_score(fio) → float  (replicates fio.py logic, no import needed)
#
# DO NOT import from fio.py here — this file must be importable standalone.
# ─────────────────────────────────────────────────────────────────────────────

import json
import re
import math
from typing import Any, Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Field registry
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (field_name, chroma_type, default, description, group)
# chroma_type: "str" | "float" | "bool" | "json_str" | "skip"
# group: "identity" | "interaction" | "optimizer" | "internal"

FIO_FIELD_REGISTRY = [
    # ── Group A: Identity & timing ────────────────────────────────────────────
    ("id",               "str",      "",    "Unique FIO ID: {date}-{start_time}-{count}",          "identity"),
    ("date",             "str",      "",    "DAY1..DAY7 — day boundary for temporal filter",        "identity"),
    ("start_time",       "str",      "",    "Raw timestamp hhmmssff — for Δt computation",          "identity"),
    ("end_time",         "str",      "",    "Raw timestamp hhmmssff — slot boundary assignment",    "identity"),
    ("video_path",       "str",      "",    "Source clip path — debug only",                        "identity"),

    # ── Group B: Core interaction (LLM-extracted) ─────────────────────────────
    ("persons",          "str",      "",    "ALL named persons, comma-sep. e.g. 'Alice, Tasha'",   "interaction"),
    ("action",           "str",      "",    "Primary verb phrase — main retrieval signal",          "interaction"),
    ("object",           "str",      "",    "Primary physical item interacted with",                "interaction"),
    ("tool",             "str",      "",    "Instrument used (different from object)",              "interaction"),
    ("app",              "str",      "",    "Digital application (Meituan, WeChat, timer...)",      "interaction"),
    ("source_location",  "str",      "",    "Where object/person was BEFORE action",               "interaction"),
    ("target_location",  "str",      "",    "Where object/person MOVED TO",                        "interaction"),
    ("speech_content",   "str",      "",    "Verbatim quoted speech — critical for HabitInsight/TM", "interaction"),
    ("attributes",       "json_str", "{}",  "Extra properties: color/qty/state as JSON string",    "interaction"),

    # ── Group C: Importance & optimizer ──────────────────────────────────────
    ("heuristic_score",  "float",    0.0,   "Field-presence heuristic (compute_heuristic_score)",  "optimizer"),
    ("importance_score", "float",    0.0,   "w(o) formula result — 0.0 for raw, computed for opt", "optimizer"),
    ("recency_score",    "float",    0.0,   "exp(-λΔt) — 0.0 for raw, computed for opt",           "optimizer"),
    ("optimized",        "bool",     False, "True if this FIO passed the optimizer keep_threshold", "optimizer"),
    ("action_sequence",  "json_str", "[]",  "Merged FIO history as JSON string — set by optimizer", "optimizer"),
]

# All field names in order
FIO_FIELDS: List[str] = [f for f, *_ in FIO_FIELD_REGISTRY]

# Fields that go into ChromaDB metadata (exclude "internal", keep all others)
CHROMADB_METADATA_FIELDS: List[str] = [
    f for f, ctype, *_ in FIO_FIELD_REGISTRY
    if ctype != "skip"
]

# Fields used for semantic embedding (the ChromaDB document string)
# Order matters: higher-signal fields first
EMBED_FIELD_ORDER = [
    "persons",          # RelationMap (125 Qs) + EntityLog who (41 Qs)
    "action",           # All 5 types — primary signal
    "object",           # EntityLog what, EventRecall
    "tool",             # EntityLog what/tool
    "app",              # EntityLog what/app
    "speech_content",   # HabitInsight SAID (19 Qs) + TaskMaster (59/63 Qs)
    "source_location",  # EntityLog where/before
    "target_location",  # EntityLog where/after
]

# ChromaDB type map for validation
_CHROMA_TYPE_MAP: Dict[str, type] = {
    "str":      str,
    "float":    float,
    "bool":     bool,
    "json_str": str,   # stored as JSON string in ChromaDB
}

# Default values by type
_DEFAULTS: Dict[str, Any] = {f: d for f, _, d, *_ in FIO_FIELD_REGISTRY}


# ══════════════════════════════════════════════════════════════════════════════
# Document string builder
# ══════════════════════════════════════════════════════════════════════════════

def build_document_string(fio: Dict) -> str:
    """
    Build the text ChromaDB embeds for semantic search.

    speech_content MUST be here — if it's only in metadata, semantic search
    for "hamburger" won't find the FIO where Tasha said "I don't want hamburgers".

    Attributes are unpacked so "color=red quantity=3" becomes searchable terms.
    """
    parts = []

    for field in EMBED_FIELD_ORDER:
        val = fio.get(field, "")
        if val and str(val).strip() not in ("", "null", "none", "{}"):
            parts.append(str(val).strip())

    # Unpack attributes JSON string into flat terms
    attrs_raw = fio.get("attributes", "{}")
    if attrs_raw and attrs_raw not in ("{}", "null", ""):
        try:
            attrs = json.loads(attrs_raw) if isinstance(attrs_raw, str) else attrs_raw
            if isinstance(attrs, dict):
                attr_terms = " ".join(
                    f"{k} {v}" for k, v in attrs.items()
                    if v not in (None, "", [], {})
                )
                if attr_terms.strip():
                    parts.append(attr_terms)
        except (json.JSONDecodeError, TypeError):
            pass

    return " ".join(filter(None, parts)).strip()


# ══════════════════════════════════════════════════════════════════════════════
# FIO validation — makes every value ChromaDB-safe
# ══════════════════════════════════════════════════════════════════════════════

def validate_fio(fio: Dict) -> Dict:
    """
    Return a ChromaDB-safe metadata dict.
    - str fields: coerce to str, replace None/null/"null" with ""
    - float fields: coerce to float, clamp 0.0-1.0 where applicable
    - bool fields: coerce to bool
    - json_str fields: ensure valid JSON string (dict/list → dumps, else "{}" or "[]")
    - Adds "document" field (the embedding text) if not already present
    """
    cleaned: Dict = {}

    for field, ctype, default, *_ in FIO_FIELD_REGISTRY:
        if ctype == "skip":
            continue
        raw = fio.get(field, default)

        if ctype == "str":
            if raw is None or str(raw).lower().strip() in ("none", "null", ""):
                cleaned[field] = ""
            else:
                cleaned[field] = str(raw).strip()

        elif ctype == "float":
            try:
                cleaned[field] = round(float(raw), 6)
            except (TypeError, ValueError):
                cleaned[field] = float(default)

        elif ctype == "bool":
            if isinstance(raw, bool):
                cleaned[field] = raw
            elif isinstance(raw, str):
                cleaned[field] = raw.lower() in ("true", "1", "yes")
            else:
                cleaned[field] = bool(raw)

        elif ctype == "json_str":
            if raw is None or raw == "" or raw == default:
                cleaned[field] = default  # "{}" or "[]"
            elif isinstance(raw, (dict, list)):
                try:
                    cleaned[field] = json.dumps(raw, ensure_ascii=False)
                except (TypeError, ValueError):
                    cleaned[field] = default
            elif isinstance(raw, str):
                # Validate it's parseable JSON
                try:
                    json.loads(raw)
                    cleaned[field] = raw
                except (json.JSONDecodeError, ValueError):
                    cleaned[field] = default
            else:
                cleaned[field] = default

    # Always include the document string for ChromaDB embedding
    if "document" not in fio:
        cleaned["document"] = build_document_string(cleaned)
    else:
        cleaned["document"] = fio["document"]

    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
# Heuristic importance score
# Mirrors compute_importance_score() from fio.py — no import needed.
# ══════════════════════════════════════════════════════════════════════════════

_LOW_SIGNAL_ACTIONS = {
    "look", "watch", "observe", "glance", "focus",
    "wait", "continue", "think", "lean", "turn", "smile", "reply",
}
_HIGH_SIGNAL_ACTIONS = {
    "pick", "pick_up", "take", "move", "place", "put", "hand", "give",
    "use", "open", "close", "order", "buy", "write", "cook", "wash",
    "install", "fix", "discuss", "mention", "ask",
}
_FIELD_WEIGHTS = {
    "object":          0.25,
    "app":             0.20,
    "attributes":      0.20,
    "target_location": 0.15,
    "tool":            0.10,
    "source_location": 0.05,
    "speech_content":  0.05,   # NEW — speech adds signal
}

def compute_heuristic_score(fio: Dict) -> float:
    """
    Field-presence heuristic. Mirrors fio.py's compute_importance_score().
    Added speech_content term vs original.
    """
    def _nv(v) -> str:
        if v is None: return ""
        s = str(v).strip().lower()
        return "" if s in ("", "none", "null", "unknown", "{}", "[]") else s

    action   = _nv(fio.get("action"))
    person   = _nv(fio.get("persons") or fio.get("person"))

    score = 0.2
    if action:                                    score += 0.20
    if person:                                    score += 0.10
    for field, w in _FIELD_WEIGHTS.items():
        if _nv(fio.get(field)):                   score += w
    if action in _HIGH_SIGNAL_ACTIONS:            score += 0.10
    if action in _LOW_SIGNAL_ACTIONS:             score -= 0.10

    return round(max(0.0, min(score, 1.0)), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Recency score  —  exp(-λ * Δt_hours)
# λ = log(2) / 24  →  half-life = 24 hours
# ══════════════════════════════════════════════════════════════════════════════

LAMBDA_DECAY = math.log(2) / 24.0   # half-life = 24 hours


def parse_hhmmss(time_str: Any) -> Optional[float]:
    """
    Parse a timestamp string into fractional hours.
    Format: hhmmssff (e.g. "14352008" → 14h 35m 20s → 14.589h)
    Returns None if unparseable.
    """
    s = re.sub(r"\D", "", str(time_str or ""))
    if len(s) < 6:
        return None
    try:
        hh = int(s[:2])
        mm = int(s[2:4])
        ss = int(s[4:6])
        return hh + mm / 60.0 + ss / 3600.0
    except (ValueError, IndexError):
        return None


def parse_day_int(day: Any) -> int:
    m = re.search(r"(\d+)", str(day or ""))
    return int(m.group(1)) if m else 0


def compute_recency_score(
    fio: Dict,
    t_max_hours: float,
    query_day_int: Optional[int] = None,
) -> float:
    """
    Compute recency score for a FIO.

    Δt = hours between fio.start_time and t_max_hours (end of collection window).

    Args:
        fio:           The FIO dict (needs "date" and "start_time").
        t_max_hours:   Reference time in hours-since-DAY1-start.
                       For same-day: end hour of that day's data.
                       For cross-day: can be end of DAY7 (full horizon).
        query_day_int: If provided, adds day offset to Δt.
                       query_day_int=7 means this FIO is being scored at DAY7 query time.

    Returns:
        float in [0.0, 1.0]. 1.0 = most recent. 0.0 = very old.
    """
    fio_day  = parse_day_int(fio.get("date"))
    fio_time = parse_hhmmss(fio.get("start_time"))

    if fio_time is None:
        return 0.0

    # Convert to hours from DAY1 start
    fio_abs_hours = (fio_day - 1) * 24.0 + fio_time

    # t_max should also be in absolute hours from DAY1 start
    # If caller passes raw hours within day, add day offset
    if t_max_hours < 24.0 and query_day_int is not None:
        # t_max is within-day hours — make absolute
        t_max_abs = (query_day_int - 1) * 24.0 + t_max_hours
    else:
        t_max_abs = t_max_hours

    delta_hours = max(0.0, t_max_abs - fio_abs_hours)
    score = math.exp(-LAMBDA_DECAY * delta_hours)
    return round(min(1.0, max(0.0, score)), 6)


def compute_recency_for_day(fios: List[Dict], day_int: int) -> Dict[str, float]:
    """
    Batch-compute recency scores for all FIOs in a day.
    Uses the last FIO's start_time as t_max (most recent clip = 1.0).

    Returns: {fio_id: recency_score}
    """
    if not fios:
        return {}

    # Find t_max = last start_time on this day
    day_fios = [f for f in fios if parse_day_int(f.get("date")) == day_int]
    if not day_fios:
        return {}

    times = [parse_hhmmss(f.get("start_time")) for f in day_fios]
    times = [t for t in times if t is not None]
    t_max = max(times) if times else 23.99  # fallback: end of day

    # Absolute t_max
    t_max_abs = (day_int - 1) * 24.0 + t_max

    scores: Dict[str, float] = {}
    for f in fios:
        fid = f.get("id", "")
        scores[fid] = compute_recency_score(f, t_max_abs)

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# FIO template — blank FIO with all defaults
# ══════════════════════════════════════════════════════════════════════════════

def blank_fio() -> Dict:
    """Return a dict with all FIO fields set to their defaults."""
    return {f: d for f, _, d, *_ in FIO_FIELD_REGISTRY}


def fio_summary(fio: Dict) -> str:
    """One-line string for logging."""
    return (
        f"[{fio.get('date','?')} {fio.get('start_time','?')}] "
        f"persons='{fio.get('persons','')}' "
        f"action='{fio.get('action','')}' "
        f"obj='{fio.get('object','')}' "
        f"speech='{(fio.get('speech_content') or '')[:40]}' "
        f"imp={fio.get('importance_score',0.0):.3f} "
        f"opt={fio.get('optimized', False)}"
    )