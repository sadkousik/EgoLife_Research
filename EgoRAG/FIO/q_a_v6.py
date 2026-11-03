#q_a_v6.py
# ─────────────────────────────────────────────────────────────────────────────
# QA pipeline — final version integrating all fixes.
#
# FIXES vs q_a_v5.py (hierarchy key/attribute mismatches):
#   HIER-1  — slot handler read non-existent "person_presence_count" /
#             "speech_highlights" → replaced with actual fields:
#             top_objects, top_actions, action_object_pairs, llm_summary
#   HIER-2  — daily handler read non-existent "key_events_timeline" /
#             "speech_of_the_day" → replaced with actual fields:
#             top_objects, top_actions, slot_breakdown, llm_summary
#   HIER-3  — HabitInsight slot path always produced empty persons;
#             now mines person names from llm_summary text
#   LOGIC-1 — EntityLog, RelationMap, EventRecall, TaskMaster never cascaded
#             to hierarchy on deterministic failure → added _try_hier_cascade()
#             called at every LLM fallback site
#   LOGIC-2 — TaskMaster now cascades when speech evidence < 2 FIOs
#
# OPT PATCHES (merged FIO attributes now used when db="opt"):
#   OPT-A  — soft_filter_fio_and_rank: e/a/importance ranking. The dynamic
#             α-table is currently disabled, so alpha_score is NOT used in the
#             score. Formula (per ABLATION mode, weights on e/a/imp only):
#               score = w_e·e + w_a·a + w_imp·imp
#   OPT-B  — action_sequence expansion: merged FIO event history unpacked
#             into individual flat evidence dicts before every LLM call.
#             Uses _parse_action_sequence() + expand_opt_evidence().
#   OPT-C  — evidence logging shows alpha_score and merged step count when
#             db="opt" via log_opt_evidence() (display only, not scoring).
# ─────────────────────────────────────────────────────────────────────────────

import json
import re
import math
import os
import datetime
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
print("before qdecompose")
import Q_decompose
print("after qdecompose")
from experiment_logger import start_experiment
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

_embed_model = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

from chroma_fio_v2 import DualFIOStorage, get_collection_name, get_where_filter


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION KNOB: number of evidence items sent to the LLM
# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE_K is the SINGLE control for how many ranked evidence items reach the
# LLM prompt. It is the only number that varies across the evidence-count
# ablation. Sweep it per run, e.g.:
#     EVIDENCE_K=1 DB_MODE=opt python q_a_v6.py
#     EVIDENCE_K=2 DB_MODE=opt python q_a_v6.py
#     ... 4, 6, 8, 12 ...
#
# Mechanics (so K is faithfully binding at every value):
#   * llm_fallback_answer() hard-caps the evidence list to EVIDENCE_K. This is
#     the single chokepoint every LLM call passes through, so the model never
#     sees more than K items regardless of which handler built the list.
#   * The evidence builders (_llm_evidence / expand_opt_evidence) and the
#     per-handler slices produce UP TO EVIDENCE_K candidates, so the cap binds
#     (raising K actually feeds more; lowering K feeds fewer) instead of being
#     masked by smaller hard-coded slices.
#   * apply_temporal_relation keeps EVIDENCE_K neighbours of context for
#     first_time / last_time questions, for the same reason.
# Use K >= 1.
EVIDENCE_K = int(os.environ.get("EVIDENCE_K", "5"))
print(f"[ABLATION] EVIDENCE_K = {EVIDENCE_K} (max evidence items per LLM call)")


# ══════════════════════════════════════════════════════════════════════════════
# Primitive helpers
# ══════════════════════════════════════════════════════════════════════════════

def normalize_date(d: str) -> int:
    if not d:
        return 0
    return int(str(d).replace("DAY", "").replace("day", "").strip() or 0)

def normalize_time(t) -> int:
    if not t:
        return 0
    return int(str(t)[:6])

def embed(text: str) -> Optional[np.ndarray]:
    if not text:
        return None
    return np.array(_embed_model([text])[0])

def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ══════════════════════════════════════════════════════════════════════════════
# Entity / action scoring
# ══════════════════════════════════════════════════════════════════════════════

SLOT_ENTITIES = {"app", "tool", "object", "person"}

def score_entity_match(query_entity: Optional[str], meta: Dict) -> float:
    if not query_entity:
        return 0.0
    qe = query_entity.lower()
    if qe in SLOT_ENTITIES or qe in {"person", "someone", "someone_else"}:
        return 0.0
    candidates = [meta.get("object", ""), meta.get("tool", ""), meta.get("app", ""), meta.get("speech_content", ""),]
    for c in candidates:
        if qe in str(c).lower():
            return 1.0
    q_emb = embed(qe)
    scores = [cosine(q_emb, embed(str(c))) for c in candidates if c]
    return max(scores) if scores else 0.0

def score_action_match(query_action: Optional[str], meta_action: Optional[str]) -> float:
    if not query_action or not meta_action:
        return 0.0
    qa = query_action.lower().strip()
    ma = meta_action.lower().strip()
    if qa in ma or ma in qa:
        return 1.0
    return cosine(embed(qa), embed(ma))


# ══════════════════════════════════════════════════════════════════════════════
# Temporal helpers
# ══════════════════════════════════════════════════════════════════════════════

def meta_time_key(meta: Dict):
    return (normalize_date(meta.get("date")), normalize_time(meta.get("start_time", 0)))

def filter_by_query_time(metas: List[Dict], question_info: Dict) -> List[Dict]:
    q_date = normalize_date(question_info.get("day"))
    q_time = normalize_time(question_info.get("time", 0))
    filtered = []
    for m in metas:
        m_date = normalize_date(m.get("date"))
        m_time = normalize_time(m.get("start_time"))
        if m_date > q_date:
            continue
        if m_date == q_date and m_time > q_time:
            continue
        filtered.append(m)
    return filtered

def apply_temporal_relation_not_used(
    metas: List[Dict],
    temporal_relation: Optional[str],
    ref_meta: Optional[Dict] = None,
) -> List[Dict]:
    if not temporal_relation:
        return metas
    metas = sorted(metas, key=meta_time_key)
    if temporal_relation == "first_time":
        return metas[:1]
    if temporal_relation == "last_time":
        return metas[-1:]
    if temporal_relation == "before" and ref_meta:
        ref_t = meta_time_key(ref_meta)
        return [m for m in metas if meta_time_key(m) < ref_t]
    if temporal_relation == "after" and ref_meta:
        ref_t = meta_time_key(ref_meta)
        return [m for m in metas if meta_time_key(m) > ref_t]
    return metas

#updated apply_temporal relation_code
def apply_temporal_relation(
    metas: List[Dict],
    temporal_relation: Optional[str],
    ref_meta: Optional[Dict] = None,
    top_n: int = EVIDENCE_K,             # how many neighbours to keep for the LLM (ablation)
) -> List[Dict]:
    if not temporal_relation:
        return metas
    metas = sorted(metas, key=meta_time_key)   # ascending by (date, time)

    if temporal_relation == "first_time":
        # earliest occurrence is the answer (ranked[0] = metas[0]),
        # keep the next few for context instead of collapsing to 1
        return metas[:top_n]

    if temporal_relation == "last_time":
        # most-recent occurrence is the answer → put it FIRST, then older context
        return list(reversed(metas[-top_n:]))

    if temporal_relation == "before" and ref_meta:
        ref_t = meta_time_key(ref_meta)
        return [m for m in metas if meta_time_key(m) < ref_t]

    if temporal_relation == "after" and ref_meta:
        ref_t = meta_time_key(ref_meta)
        return [m for m in metas if meta_time_key(m) > ref_t]

    return metas
##updated apply_temporal relation_code [end]

# ══════════════════════════════════════════════════════════════════════════════
# Relevance-aware candidate pool for temporal questions
#   FIX: first_time / last_time must take the extremum OF THE QUERIED ENTITY,
#   not the chronologically-extreme FIO in the whole retrieved pool. Each filter
#   is applied only if it leaves a non-empty result, so it degrades gracefully.
# ══════════════════════════════════════════════════════════════════════════════

_ACTION_SYN = {
    "buy":  {"buy", "bought", "buying", "purchase", "purchased", "got", "get"},
    "use":  {"use", "used", "using"},
    "take": {"take", "took", "taking", "picked", "pick"},
}

def _action_group(a: Optional[str]):
    al = (a or "").lower()
    for g in _ACTION_SYN.values():
        if any(w in al for w in g):
            return g
    return None

def _field_contains(meta: Dict, needle: Optional[str], fields) -> bool:
    nl = str(needle or "").lower().strip()
    if not nl:
        return False
    nl_ns = nl.replace(" ", "")          # "hot pot" -> "hotpot"
    for f in fields:
        hay = str(meta.get(f, "")).lower()
        if nl in hay:
            return True
        if len(nl_ns) >= 4 and nl_ns in hay.replace(" ", ""):
            return True
    return False

# communication-only actions: a "mention"/"said" is not a physical use of a thing
_COMM_ACTIONS = {
    "said", "say", "says", "mentioned", "mention", "asked", "ask", "replied",
    "reply", "suggested", "commented", "comment", "talked", "told", "tell",
    "encouraged", "repeated", "noted", "responded", "answered", "explained",
}
# tokens stripped from a "before <event>" anchor clause to leave its content words
_CTX_STOP = {
    "i", "we", "he", "she", "they", "you", "the", "a", "an", "out", "up", "off",
    "took", "take", "taken", "taking", "get", "got", "getting", "put", "this",
    "that", "my", "your", "our", "it", "of", "to", "was", "were", "is", "am",
    "are", "had", "have", "has", "do", "did", "down", "over", "with",
}

def _meta_time_key(m: Dict) -> Tuple[int, float]:
    return (normalize_date(m.get("date", "DAY1")),
            parse_hhmmss_to_mins(m.get("start_time", 0)) or 0.0)

def temporal_candidate_pool(
    metas: List[Dict],
    q_json: Dict,
    question_text: str,
    query_day_n: int,
) -> List[Dict]:
    pool = list(metas)
    qtext_l = (question_text or "").lower()

    # 1) entity — literal presence in object/tool/app/speech/action/location.
    #    Location is included so "cooked food, loc='on the grill'" matches the
    #    entity "grill" (the grill-use FIO would otherwise be dropped).
    entity = q_json.get("entity")
    if entity and str(entity).lower() not in SLOT_ENTITIES:
        efields = ("object", "tool", "app", "speech_content", "action",
                   "source_location", "target_location", "location")
        ent = [m for m in pool if _field_contains(m, entity, efields)]
        if not ent:                                   # fall back to embedding
            ent = [m for m in pool if score_entity_match(entity, m) >= 0.55]
        if ent:
            pool = ent

    # 1b) for a PHYSICAL query ("used"/"played"/"bought"...) drop pure speech
    #     "mentions" — talking about a grill is not using the grill. Skipped when
    #     the query itself is about talking ("when did we talk about coffee").
    q_act = str(q_json.get("action") or "").lower()
    q_is_comm = any(w in q_act for w in
                    ("talk", "said", "say", "mention", "ask", "discuss", "tell"))
    if q_act and not q_is_comm:
        non_comm = [m for m in pool
                    if str(m.get("action") or "").lower().strip() not in _COMM_ACTIONS]
        if non_comm:
            pool = non_comm

    # 2) action — keep FIOs whose action is compatible, if any are
    action = q_json.get("action")
    if action:
        grp = _action_group(action)
        act = [m for m in pool
               if score_action_match(action, m.get("action")) >= 0.5
               or (grp is not None and _action_group(m.get("action")) is grp)]
        if act:
            pool = act

    # 3) related person — keep that person's FIOs, if any
    rel = q_json.get("related_person")
    if rel:
        rl = str(rel).lower().strip()
        per = [m for m in pool if rl in str(m.get("person", "")).lower()
               or rl in str(m.get("persons", "")).lower()]
        if per:
            pool = per

    # 4) "today" / "this <part-of-day>" → restrict to the query day, if any survive
    if re.search(r"\btoday\b|\bthis (morning|afternoon|evening|noon)\b", qtext_l):
        same = [m for m in pool if normalize_date(m.get("date")) == query_day_n]
        if same:
            pool = same

    # 5) contextual "before <event>" anchor → the answer is the last/first
    #    occurrence strictly BEFORE that event. Resolve the anchor's time from
    #    the full retrieved set and cap the pool to events preceding it.
    mb = re.search(r"\bbefore\b\s+(?:i|we|he|she|they|you)?\s*(.+?)[,?]", qtext_l)
    if mb:
        anchor_terms = [t for t in re.findall(r"[a-z]+", mb.group(1))
                        if t not in _CTX_STOP and len(t) > 2]
        if anchor_terms:
            anchors = [m for m in metas if any(
                t in " ".join(str(m.get(f, "")).lower()
                              for f in ("object", "action", "speech_content"))
                for t in anchor_terms)]
            if anchors:
                cap = max(_meta_time_key(a) for a in anchors)
                capped = [m for m in pool if _meta_time_key(m) < cap]
                print(f"  [before-anchor] terms={anchor_terms} cap<{cap} "
                      f"| kept {len(capped)}/{len(pool)}")
                if capped:
                    pool = capped

    return pool


# ══════════════════════════════════════════════════════════════════════════════
# JSON extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_qwen_json(text: str) -> dict:
    if not text:
        raise ValueError("Empty response")
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON found")
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                raw_json = text[start:i + 1]
                break
    else:
        raise ValueError("Incomplete JSON")
    raw_json = re.sub(r'"([^"]+)"\s*\|\s*"([^"]+)"', r'["\1", "\2"]', raw_json)
    raw_json = re.sub(r',\s*}', '}', raw_json)
    raw_json = re.sub(r',\s*]', ']', raw_json)
    return json.loads(raw_json)


# ══════════════════════════════════════════════════════════════════════════════
# MCQ answer formatting
# ══════════════════════════════════════════════════════════════════════════════

def format_mcq_answer(option: Optional[str], explanation: str) -> str:
    if option not in {"A", "B", "C", "D"}:
        return "Answer: Cannot be determined."
    return f"Answer: {option}\nExplain your choice in one sentence. {explanation}"

# def extract_single_option_answer(answer: Optional[str]) -> Optional[str]:
#     if not answer:
#         return None
#     match = re.search(r"\b([A-D])\b", answer)
#     return match.group(1) if match else None
def extract_single_option_answer(
    answer: Optional[str],
    question: Optional[Dict] = None,
) -> Optional[str]:
    if not answer:
        return None

    # letter the model actually emitted (prefer the one after "Answer:")
    m = re.search(r"answer\s*[:\-]?\s*([A-D])\b", answer, re.IGNORECASE)
    if not m:
        m = re.search(r"\b([A-D])\b", answer)
    emitted = m.group(1).upper() if m else None

    if question is None:
        return emitted

    # which choice TEXT does the explanation actually point at?
    low = answer.lower()
    content_opt, best_len = None, 0
    for c in "ABCD":
        text = (question.get(f"choice_{c.lower()}") or "").lower().strip()
        if text and text in low and len(text) > best_len:
            best_len, content_opt = len(text), c

    # prefer the content-grounded option when it disagrees with the bare letter
    if content_opt and content_opt != emitted:
        print(f"  [extract] emitted={emitted} but explanation names choice "
              f"{content_opt} ('{question.get(f'choice_{content_opt.lower()}')}') "
              f"→ overriding to {content_opt}")
        return content_opt

    return emitted

# ══════════════════════════════════════════════════════════════════════════════
# Where answer resolution
# ══════════════════════════════════════════════════════════════════════════════

def resolve_where_answer(meta: Optional[Dict], temporal_relation: Optional[str] = None) -> Optional[str]:
    if not meta:
        return None
    src = meta.get("source_location")
    tgt = meta.get("target_location")
    if temporal_relation in {"before", "originally", "first_time"}:
        return src or tgt
    if temporal_relation in {"after", "last_time"}:
        return tgt or src
    return tgt or src


# ══════════════════════════════════════════════════════════════════════════════
# LLM fallback
# ══════════════════════════════════════════════════════════════════════════════

def llm_fallback_answer(
    question: Dict[str, Any],
    qwen_q_json: Dict,
    ranked: List[Dict],
    question_info: Dict,
) -> str:
    # ── ABLATION: hard-cap evidence count (single chokepoint for ALL callers) ──
    # Every LLM call in this pipeline routes through here, so capping `ranked`
    # to EVIDENCE_K guarantees the model never sees more than K evidence items,
    # whatever handler assembled the list. The print line makes the effective
    # count auditable per question / per question-type for the ablation.
    _n_in = len(ranked)
    if _n_in > EVIDENCE_K:
        ranked = ranked[:EVIDENCE_K]
    print(f"  [evidence-ablation] type={question.get('type','?')} "
          f"K={EVIDENCE_K} | received={_n_in} | used={len(ranked)}")

    answer_prompt = f"""
Given the decomposed question: {json.dumps(qwen_q_json)},
and the query intent: {qwen_q_json.get("query_intent")},
and the question info: {json.dumps(question_info)},
and question info contains choices:
A. {question.get("choice_a")}
B. {question.get("choice_b")}
C. {question.get("choice_c")}
D. {question.get("choice_d")}

Using the following first-person video evidence,
infer the most plausible answer.

Evidence (ranked):
{json.dumps(ranked, indent=2, default=str)}

Return response in one sentence and choose from A, B, C, D only.
Use the format:
Answer: <A/B/C/D>
Explain your choice in one sentence.
"""
    try:
        from Q_decompose import tokenizer as _fb_tok
        _fb_messages = [
            {"role": "system", "content": Q_decompose.system_message},
            {"role": "user",   "content": answer_prompt},
        ]
        _fb_prompt_str  = _fb_tok.apply_chat_template(
            _fb_messages, tokenize=False, add_generation_prompt=True
        )
        _fb_inputs      = _fb_tok(_fb_prompt_str, return_tensors="pt")
        _fb_input_tokens = _fb_inputs["input_ids"].shape[1]
    except Exception:
        _fb_input_tokens = len(answer_prompt) // 4

    n_evidence = len(ranked)
    print(f"  [LLM fallback] evidence={n_evidence} items | "
          f"input_tokens={_fb_input_tokens} | "
          f"context_chars={len(answer_prompt)}")

    answer = Q_decompose.call_qwen(
        prompt=answer_prompt,
        system_message=Q_decompose.system_message,
    )

    _fb_output_tokens = len(answer) // 4 if answer else 0
    print(f"  [LLM fallback] output_tokens={_fb_output_tokens} | "
          f"total_tokens={_fb_input_tokens + _fb_output_tokens}")

    return answer or "Answer: Cannot be determined."


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

PERSON_NAMES = {
    "Alice", "Tasha", "Shure", "Lucia", "Katrina",
    "Nicous", "Jake", "Choiszt", "Violet", "Bob",
}
ME_TOKENS = {"me", "mine", "i", "myself"}

CAMERA_WEARER_NAME: str = "Jake"
ME_CHOICE_TOKENS = ME_TOKENS | {"jake"}

def set_camera_wearer(name: str):
    global CAMERA_WEARER_NAME, ME_CHOICE_TOKENS, PERSON_NAMES
    CAMERA_WEARER_NAME = name.strip().capitalize()
    ME_CHOICE_TOKENS = ME_TOKENS | {CAMERA_WEARER_NAME.lower()}
    PERSON_NAMES.add(CAMERA_WEARER_NAME)
    print(f"[Config] Camera wearer = '{CAMERA_WEARER_NAME}' "
          f"| Me-tokens = {ME_CHOICE_TOKENS}")

WHEN_LABEL_RANGES = {
    "About 10 minutes ago":              (5,    20),
    "About 15 minutes ago":              (10,   25),
    "About 20 minutes ago":              (15,   30),
    "About half an hour ago":            (20,   45),
    "About 40 minutes ago":              (30,   55),
    "About 50 minutes ago":              (40,   65),
    "About an hour ago":                 (45,   90),
    "About an hour and a half ago":      (80,   110),
    "About 2 hours ago":                 (100,  140),
    "About three hours ago":             (140,  220),
    "About four hours ago":              (200,  280),
    "About four and a half hours ago":   (250,  310),
    "About six hours ago":               (300,  420),
    "About eight hours ago":             (420,  560),
    "About 12 hours ago":                (600,  780),
    "This morning":                      (60,   480),
    "This afternoon":                    (240,  600),
    "Today at noon":                     (300,  480),
    "Last night":                        (660,  1200),
    "One day ago":                       (1320, 1560),
    "About a day ago":                   (1380, 1560),
    "1 day 6 hours ago":                 (1700, 1900),
    "Yesterday morning":                 (1320, 1920),
    "Yesterday at noon":                 (1320, 2040),
    "Yesterday afternoon":               (1100, 1560),
    "The night before yesterday":        (2400, 3000),
    "The day before yesterday":          (2400, 3000),
    "The afternoon before yesterday":    (2700, 2940),
    "The day before yesterday at noon":  (3120, 3360),
    "About two days ago":                (2700, 2940),
    "Two days ago":                      (2700, 3000),
    "Three days ago":                    (4200, 4500),
    "Three days ago at noon":            (4440, 4680),
    "Four days ago":                     (5760, 6120),
}


# ══════════════════════════════════════════════════════════════════════════════
# OPT-A: action_sequence helpers
# ══════════════════════════════════════════════════════════════════════════════

def _parse_action_sequence(fio: Dict) -> List[Dict]:
    """Parse action_sequence JSON string. Returns [] if absent/invalid."""
    seq_str = fio.get("action_sequence", "[]")
    if not seq_str or seq_str in ("[]", "", "null"):
        return []
    try:
        seq = json.loads(seq_str)
        return seq if isinstance(seq, list) else []
    except (json.JSONDecodeError, TypeError):
        return []

def _has_merged_fios(ranked: List[Dict]) -> bool:
    """True if any FIO in ranked has a non-empty action_sequence."""
    return any(_parse_action_sequence(f) for f in ranked)

def expand_opt_evidence(ranked: List[Dict], max_evidence: int = EVIDENCE_K) -> List[Dict]:
    """
    OPT-B: Expand merged opt FIOs into individual evidence items for the LLM.

    For each FIO:
      - Unmerged FIO (action_sequence == []): include as-is.
      - Merged FIO (action_sequence non-empty): include canonical FIO PLUS
        each step in action_sequence as a separate flat evidence dict.

    This ensures the LLM sees every compressed event, not just the canonical
    action/object of the merged FIO. Capped at max_evidence total items.
    """
    expanded = []
    for fio in ranked:
        if len(expanded) >= max_evidence:
            break
        steps = _parse_action_sequence(fio)

        canonical = {
            "action":           fio.get("action", ""),
            "object":           fio.get("object", ""),
            "tool":             fio.get("tool", ""),
            "app":              fio.get("app", ""),
            "persons":          fio.get("persons", ""),
            "speech_content":   fio.get("speech_content", ""),
            "source_location":  fio.get("source_location", ""),
            "target_location":  fio.get("target_location", ""),
            "date":             fio.get("date", ""),
            "start_time":       fio.get("start_time", ""),
            "end_time":         fio.get("end_time", ""),
            "importance_score": fio.get("importance_score", 0.0),
            "alpha_score":      fio.get("alpha_score", 0.5),
            "source":           "opt_fio_canonical",
        }
        expanded.append(canonical)

        for step in steps:
            if len(expanded) >= max_evidence:
                break
            if not isinstance(step, dict):
                continue
            step_ev = {
                "action":           step.get("action", ""),
                "object":           step.get("object", ""),
                "speech_content":   step.get("speech_content", ""),
                "date":             fio.get("date", ""),
                "start_time":       step.get("start_time", fio.get("start_time", "")),
                "end_time":         step.get("end_time",   fio.get("end_time",   "")),
                "importance_score": fio.get("importance_score", 0.0),
                "alpha_score":      fio.get("alpha_score", 0.5),
                "source":           "opt_fio_merged_step",
            }
            attrs_str = step.get("attributes", "{}")
            try:
                attrs = json.loads(attrs_str) if isinstance(attrs_str, str) else {}
                if attrs:
                    step_ev["attributes"] = attrs_str
            except (json.JSONDecodeError, TypeError):
                pass
            expanded.append(step_ev)

    return expanded[:max_evidence]


# ══════════════════════════════════════════════════════════════════════════════
# OPT-A: opt-aware soft ranking
# ══════════════════════════════════════════════════════════════════════════════

def soft_filter_fio_and_rank(
    metas: List[Dict],
    q_json: Dict,
    db: str = "raw",
) -> List[Dict]:
    """
    Re-ranking with db-aware scoring.

    Score formula (e / a / importance only):
        score = w_e · e_score + w_a · a_score + w_imp · importance_score

    The α-table (alpha_score) is NOT used in ranking — the dynamic
    interaction-strength table is currently disabled, so alpha_score is
    excluded from the score. Weights are chosen per ABLATION mode below.
    importance_score and the e/a match scores are the only ranking signals.
    """
    scored = []
    for meta in metas:
        if not meta.get("action") and not meta.get("object"):
            continue

        e_score = score_entity_match(q_json.get("entity"), meta)
        if q_json.get("query_intent") == "what":
            e_score = max(e_score, 0.2)

        a_score = score_action_match(q_json.get("action"), meta.get("action", ""))
        imp     = float(meta.get("importance_score") or 0.0)
        # alpha_score intentionally NOT read — dynamic α-table is disabled,
        # so it does not participate in ranking.

        # Weights are (w_e, w_a, w_imp). alpha is no longer a ranking signal.
        weights = {
            "full":     (0.53, 0.29, 0.18),
            "no_e":     (0.00, 0.62, 0.38),
            "no_a":     (0.75, 0.00, 0.25),
            "no_imp":   (0.65, 0.35, 0.00),#(0.50, 0.50, 0.00) #original (0.65, 0.35, 0.00)
            "equal":    (0.34, 0.33, 0.33),
            "only_e":   (1.00, 0.00, 0.00),
        }[ABLATION]

        w_e, w_a, w_imp = weights
        score = w_e*e_score + w_a*a_score + w_imp*imp

        if score > 0.01:
            scored.append((score, meta))

    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]


# ══════════════════════════════════════════════════════════════════════════════
# OPT-C: evidence logging
# ══════════════════════════════════════════════════════════════════════════════

def _log_evidence(ranked: List[Dict], db: str = "raw") -> None:
    """Print top-5 evidence with alpha_score and merged step count when db=opt."""
    if not ranked:
        print('  [Evidence] no FIOs retrieved — will cascade to hierarchy')
        return
    top5 = ranked[:5]
    print(f'  [Evidence] top-{len(top5)} FIOs:')
    for ei, m in enumerate(top5):
        persons = m.get('persons', '') or '—'
        action  = m.get('action',  '') or '—'
        obj     = m.get('object',  '') or '—'
        speech  = (m.get('speech_content', '') or '')[:40]
        loc_s   = m.get('source_location', '') or ''
        imp     = float(m.get('importance_score') or 0.0)
        date    = m.get('date', '')
        t       = str(m.get('start_time', ''))[:6]
        line    = (f'    [{ei}] {date} {t} | p={persons} | a={action} '
                   f'| o={obj} | w={imp:.3f}')
        if db == "opt":
            alpha   = float(m.get('alpha_score') or 0.5)
            n_steps = len(_parse_action_sequence(m))
            line   += f' | α={alpha:.3f}'
            if n_steps:
                line += f' | merged={n_steps}'
        if speech: line += f" | speech='{speech}'"
        if loc_s:  line += f' | loc={loc_s}'
        print(line)


# ══════════════════════════════════════════════════════════════════════════════
# Person extraction helpers
# ══════════════════════════════════════════════════════════════════════════════

def extract_persons_from_meta(meta: Dict) -> List[str]:
    persons_str = meta.get("persons") or meta.get("person") or ""
    if not persons_str:
        return []
    result = []
    for p in re.split(r"[,/&]| and ", str(persons_str)):
        p = p.strip()
        if not p:
            continue
        if p.lower() == CAMERA_WEARER_NAME.lower():
            result.append("Me")
        elif p in PERSON_NAMES:
            result.append(p)
    return result

def resolve_person(meta: Dict, question: Dict) -> Optional[str]:
    persons = extract_persons_from_meta(meta)
    if persons:
        return persons[0]
    q_lower = question.get("question", "").lower()
    POSSESSION_SIGNALS = [
        "in my hand", "in my ", "whose phone", "whose keyboard",
        "whose bag", "whose cup", "whose bottle", "whose book",
        "i am holding", "i was holding",
    ]
    if any(s in q_lower for s in POSSESSION_SIGNALS):
        choices = {c: question.get(f"choice_{c.lower()}", "") for c in ["A", "B", "C", "D"]}
        for opt, text in choices.items():
            if text.lower().strip() in ME_CHOICE_TOKENS:
                return text.strip()
    return None


# ── qualifier-aware who resolution ───────────────────────────────────────────
# FIX: a who-question can carry a disambiguating location/object detail
# ("plate ... next to the table", "plate of strawberries"). Picking ranked[0]
# by semantic score alone ignores it; re-rank candidates by how well their
# loc/object matches those qualifiers, but only override when one actually hits.
_QUAL_STOPWORDS = {
    "the", "a", "an", "of", "to", "next", "near", "by", "beside", "on", "in",
    "at", "this", "that", "my", "your", "our", "with", "and", "is", "was",
    "were", "it", "side", "top",
}

def _question_qualifiers(question_text: str, q_json: Dict) -> set:
    s = (question_text or "").lower()
    loc_terms = set(re.findall(
        r"(?:next to|near|beside|on|in|at)\s+(?:the\s+|a\s+)?([a-z]+)", s))
    obj_terms = set(re.findall(r"\bof\s+([a-z]+)", s))
    ent = str(q_json.get("entity") or "").lower()
    return {t for t in (loc_terms | obj_terms)
            if t and t not in _QUAL_STOPWORDS and t != ent}

def _fio_qualifier_score(meta: Dict, terms: set) -> float:
    if not terms:
        return 0.0
    hay = " ".join(str(meta.get(f, "")).lower() for f in (
        "object", "speech_content", "source_location",
        "target_location", "location"))
    return sum(1.0 for t in terms if t in hay)

def who_answer(ranked: List[Dict], question: Dict, q_json: Dict) -> Optional[str]:
    terms = _question_qualifiers(question.get("question", ""), q_json)
    if not terms or not ranked:
        return resolve_person(ranked[0], question) if ranked else None
    best, best_q, best_s = None, 0.0, -1.0
    for m in ranked[:10]:
        if not extract_persons_from_meta(m):
            continue
        q = _fio_qualifier_score(m, terms)
        s = (q
             + 0.5 * score_entity_match(q_json.get("entity"), m)
             + 0.5 * score_action_match(q_json.get("action"), m.get("action")))
        if s > best_s:
            best_s, best_q, best = s, q, m
    # trust the qualifier-based pick only if a location/object term actually matched
    if best is not None and best_q > 0:
        return resolve_person(best, question)
    return resolve_person(ranked[0], question)


# ══════════════════════════════════════════════════════════════════════════════
# WHEN handler
# ══════════════════════════════════════════════════════════════════════════════

def parse_hhmmss_to_mins(time_val: Any) -> Optional[float]:
    s = re.sub(r"\D", "", str(time_val or ""))
    if len(s) < 6:
        return None
    try:
        h = int(s[:2]); m = int(s[2:4])
        return h * 60.0 + m
    except (ValueError, IndexError):
        return None

def _mins_to_human(mins: int) -> str:
    if mins < 2:     return "just now"
    if mins < 60:    return f"about {mins} minutes ago"
    if mins < 120:   return f"about {mins//60} hour ago"
    if mins < 1440:  return f"about {mins//60} hours ago"
    if mins < 2880:  return "about 1 day ago"
    return f"about {mins//1440} days ago"

# ── calendar-relative answer mapping ─────────────────────────────────────────
# FIX: options like "Last night" / "The afternoon before yesterday" / "Two days
# ago" cannot be matched by a fixed minute window (the gap depends on the
# time-of-day of both the event and the query). For previous-day answers we key
# on day-offset + time-of-day bucket; same-day answers stay minute-based.

def _time_bucket(mins: float) -> str:
    h = int((mins // 60) % 24)
    if h >= 21 or h < 5:  return "night"
    if 5 <= h < 11:       return "morning"
    if 11 <= h < 13:      return "noon"
    if 13 <= h < 17:      return "afternoon"
    return "evening"

_BUCKET_ADJ = {("evening", "night"), ("night", "evening"),
               ("noon", "afternoon"), ("afternoon", "noon"),
               ("noon", "morning"), ("morning", "noon")}

def _option_calendar(text: str):
    """(day_offset, bucket|None) implied by a calendar-style option, else None."""
    s = text.lower()
    if ("minute" in s or "hour" in s) and not any(
            w in s for w in ("yesterday", "day ago", "days ago", "last night")):
        return None
    bucket = next((b for b in ("morning", "noon", "afternoon", "evening", "night")
                   if b in s), None)
    if   "before yesterday" in s or "two days" in s: off = 2
    elif "three days" in s:                          off = 3
    elif "four days" in s:                           off = 4
    elif "last night" in s:                          off, bucket = 1, "night"
    elif "yesterday" in s or "one day" in s or "a day ago" in s: off = 1
    elif "today" in s or "this " in s:               off = 0
    else:
        return None
    return (off, bucket)

def _minute_range_score(text: str, delta: float) -> float:
    cl = text.lower().strip()
    for label, (lo, hi) in WHEN_LABEL_RANGES.items():
        if label.lower() in cl or cl in label.lower():
            if lo <= delta <= hi:
                return 2.0
            mid = (lo + hi) / 2.0
            return 1.0 / (1.0 + abs(delta - mid) / max(hi - lo, 1.0))
    return 0.0

_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "half": 0.5,
}

def _parse_relative_minutes(text: str) -> Optional[float]:
    """Minutes implied by a same-day option, parsed generically so the option
    set isn't limited to the hand-maintained WHEN_LABEL_RANGES table. Returns
    None for non-relative phrasings (handled elsewhere)."""
    s = text.lower()
    if "just now" in s or "moment ago" in s:
        return 0.0
    if "half an hour" in s or "half hour" in s:
        return 30.0
    # "<n> and a half hours"
    mh = re.search(r"(\d+|[a-z]+)\s+and\s+a\s+half\s+hour", s)
    if mh:
        n = _NUM_WORDS.get(mh.group(1))
        if n is None:
            try: n = float(mh.group(1))
            except ValueError: n = None
        if n is not None:
            return (n + 0.5) * 60.0
    m = re.search(r"(\d+|" + "|".join(_NUM_WORDS) + r")\s*(hour|hr|minute|min)", s)
    if m:
        n = _NUM_WORDS.get(m.group(1))
        if n is None:
            try: n = float(m.group(1))
            except ValueError: n = None
        if n is not None:
            return n * 60.0 if m.group(2).startswith(("hour", "hr")) else n
    return None

def handle_when(ranked: List[Dict], question: Dict, query_info: Dict) -> Tuple[Optional[str], int]:
    if not ranked:
        return None, 0
    target   = ranked[0]
    q_day_n  = normalize_date(query_info.get("day", "DAY1"))
    q_mins   = parse_hhmmss_to_mins(query_info.get("time", 0)) or 0.0
    t_day_n  = normalize_date(target.get("date", "DAY1"))
    t_mins   = parse_hhmmss_to_mins(target.get("start_time", 0)) or 0.0
    delta_mins = max(0.0, (q_day_n - t_day_n) * 1440 + (q_mins - t_mins))

    day_off  = max(0, q_day_n - t_day_n)
    t_bucket = _time_bucket(t_mins)
    print(f"  [when] target={target.get('date')} {str(target.get('start_time'))[:6]} "
          f"| day_off={day_off} bucket={t_bucket} delta={int(delta_mins)}min")

    choices = {c: question.get(f"choice_{c.lower()}", "") for c in ["A", "B", "C", "D"]}
    best_opt, best_score = None, -1.0
    for opt, text in choices.items():
        if not text:
            continue
        cal = _option_calendar(text)
        s   = text.lower().strip()
        if day_off == 0:                       # answer is earlier the same day
            if cal is not None and cal[0] >= 1:
                score = 0.05                   # multi-day option, almost certainly wrong
            else:
                rel = _parse_relative_minutes(text)
                if rel is not None:            # "just now" / "about N hours/minutes ago"
                    score = 2.0 / (1.0 + abs(delta_mins - rel) / 30.0)
                else:
                    score = _minute_range_score(text, delta_mins)
                    if cal is not None and cal[1] and cal[1] == t_bucket:
                        score = max(score, 1.5)    # "this morning/afternoon/noon"
        else:                                  # answer is on a previous day
            if cal is None or cal[0] == 0:
                score = 0.05
            else:
                off, bk = cal
                score = 2.0 if off == day_off else (0.8 if abs(off - day_off) == 1 else 0.1)
                if   bk is None:                    score += 0.5
                elif bk == t_bucket:                score += 1.0
                elif (bk, t_bucket) in _BUCKET_ADJ: score += 0.5
        if score > best_score:
            best_score, best_opt = score, opt

    if best_opt is not None and best_score >= 0.35:
        return best_opt, int(delta_mins)
    approx = f"{int(delta_mins)} minutes ago"
    return map_answer_to_mcq_option_v2(approx, question, min_score=0.25), int(delta_mins)


# ══════════════════════════════════════════════════════════════════════════════
# MCQ option mapping
# ══════════════════════════════════════════════════════════════════════════════

def map_answer_to_mcq_option_v2(
    answer_text: str,
    question: Dict,
    min_score: float = 0.45,
) -> Optional[str]:
    if not answer_text:
        return None
    INVALID = {"unknown", "someone", "someone else", "they",
               "one of the friends", "person", "people", ""}
    if answer_text.lower().strip() in INVALID:
        return None
    answer_lower = answer_text.lower().strip()
    choices = {c: question.get(f"choice_{c.lower()}", "") for c in ["A", "B", "C", "D"]}
    if answer_lower in ME_CHOICE_TOKENS:
        for opt, text in choices.items():
            if text.lower().strip() in ME_CHOICE_TOKENS:
                return opt
    best_score, best_opt = -1.0, None
    ans_emb = embed(answer_lower)
    for opt, text in choices.items():
        if not text:
            continue
        text_l = text.lower().strip()
        if answer_lower == text_l:
            return opt
        if answer_lower in ME_CHOICE_TOKENS and text_l in ME_CHOICE_TOKENS:
            return opt
        lexical  = 1.0 if (answer_lower in text_l or text_l in answer_lower) else 0.0
        semantic = cosine(ans_emb, embed(text_l))
        score    = 0.4 * lexical + 0.6 * semantic
        if score > best_score:
            best_score = score; best_opt = opt
    return best_opt if best_score >= min_score else None


# ══════════════════════════════════════════════════════════════════════════════
# Contextual pre-processor
# ══════════════════════════════════════════════════════════════════════════════

def detect_contextual(question_text: str) -> Tuple[bool, Optional[str], str]:
    q = question_text.strip()
    first_word = q.split()[0].rstrip(",") if q.split() else ""
    if first_word in PERSON_NAMES:
        m = re.search(r",\s+([Ww]hat|[Ww]ho|[Ww]here|[Ww]hen|[Ww]hy|[Hh]ow|[Ww]hich)", q)
        if m:
            return True, q[:m.start()], q[m.start() + 2:]
    if first_word == "Now":
        m = re.search(r",\s+([Ww]hat|[Ww]ho|[Ww]here|[Ww]hen|[Ww]hy|[Hh]ow|[Ww]hich)", q)
        if m:
            return True, q[:m.start()], q[m.start() + 2:]
    if first_word == "The":
        m = re.search(r"\.\s+([Ww]hat|[Ww]ho|[Ww]here|[Ww]hen|[Ww]hy|[Hh]ow)", q)
        if m:
            return True, q[:m.start()], q[m.start() + 2:]
    return False, None, question_text


# ══════════════════════════════════════════════════════════════════════════════
# RelationMap co-occurrence handler
# ══════════════════════════════════════════════════════════════════════════════

def handle_cooccurrence(ranked: List[Dict], q_json: Dict, question: Dict) -> Optional[str]:
    q_text     = question.get("question", "")
    is_who_else = "who else" in q_text.lower() or "besides" in q_text.lower()
    named_in_q  = {p for p in PERSON_NAMES if p.lower() in q_text.lower()}
    person_counts: Counter = Counter()
    for meta in ranked:
        for p in extract_persons_from_meta(meta):
            if not (is_who_else and p in named_in_q):
                person_counts[p] += 1
        seq_raw = meta.get("action_sequence", "")
        if seq_raw and seq_raw not in ("[]", ""):
            try:
                seq = json.loads(seq_raw) if isinstance(seq_raw, str) else seq_raw
                if isinstance(seq, list):
                    for step in seq:
                        step_text = json.dumps(step).lower()
                        for p in PERSON_NAMES:
                            if p.lower() in step_text and not (is_who_else and p in named_in_q):
                                person_counts[p] += 0.5
            except (json.JSONDecodeError, TypeError):
                pass
    if not person_counts:
        choices_lower = [question.get(f"choice_{c}", "").lower() for c in ["a","b","c","d"]]
        if any(c in ME_CHOICE_TOKENS for c in choices_lower):
            return map_answer_to_mcq_option_v2("Me", question, min_score=0.3)
        return None
    choices = {c: question.get(f"choice_{c.lower()}", "") for c in ["A", "B", "C", "D"]}
    best_opt, best_score = None, -1.0
    for opt, choice_text in choices.items():
        if not choice_text:
            continue
        choice_persons = {p for p in PERSON_NAMES | {"Me", "I"}
                          if p.lower() in choice_text.lower()}
        if not choice_persons:
            score = cosine(embed(person_counts.most_common(1)[0][0].lower()),
                           embed(choice_text.lower())) if person_counts else 0.0
        else:
            evidence = set(person_counts.keys())
            overlap  = len(choice_persons & evidence)
            total    = len(choice_persons | evidence)
            score    = overlap / total if total else 0.0
        if score > best_score:
            best_score = score; best_opt = opt
    return best_opt if best_score >= 0.25 else None


# ── conditioned co-occurrence (action / anchor-person / first-person aware) ───
# FIX: the plain handler tallies every person across the whole day's retrieval,
# ignoring WHO the asker did the thing WITH and WHAT the action was. This one
# (A) resolves "who gave/handed X" to the giver, (B) anchors on the related
# person's co-presence, (C) else conditions on the action, and falls back to
# "Me" when only the anchor person appears (a first-person event).
_GIVE_VERBS  = {"gave", "give", "handed", "hand", "passed", "pass",
                "offered", "offer", "lent", "brought"}
_CLEAN_VERBS = {"clean", "cleaning", "cleaned", "wash", "washing", "washed",
                "rinse", "rinsed", "wipe", "tidy"}

def _persons_in(meta: Dict) -> set:
    ps = set(extract_persons_from_meta(meta))     # maps camera-wearer -> "Me"
    raw = str(meta.get("persons") or meta.get("person") or "").lower().strip()
    if raw in {"i", "me", "myself"} or re.search(r"\b(i|me|myself)\b", raw):
        ps.add("Me")
    return ps

def handle_cooccurrence_v2(
    ranked: List[Dict], q_json: Dict, question: Dict, top_k: int = 10
) -> Optional[str]:
    ql     = question.get("question", "").lower()
    action = str(q_json.get("action") or "").lower()
    rel    = str(q_json.get("related_person") or "").strip()
    if rel.lower() in ME_CHOICE_TOKENS:               # "me" is the asker, not an anchor
        rel = ""
    is_who_else = "who else" in ql or "besides" in ql
    named_in_q  = {p for p in PERSON_NAMES if p.lower() in ql}
    pool = ranked[:top_k] if len(ranked) > top_k else list(ranked)

    # (A) "who gave/handed X to <person>" → the AGENT of the give event
    if action and any(v in action for v in _GIVE_VERBS):
        gv = [m for m in pool
              if any(v in str(m.get("action", "")).lower() for v in _GIVE_VERBS)]
        gv.sort(key=lambda m: float(m.get("importance_score") or 0), reverse=True)
        for m in gv:
            agents = [p for p in _persons_in(m) if p.lower() != rel.lower()]
            if agents:
                opt = map_answer_to_mcq_option_v2(agents[0], question, min_score=0.35)
                if opt:
                    return opt

    # (B) anchor person dominates: keep FIOs where the anchor is co-present
    if rel:
        rp = [m for m in pool if rel.lower() in
              str(m.get("persons") or m.get("person") or "").lower()]
        if rp:
            pool = rp
    # (C) no anchor → condition on the action (incl. a clean/wash group)
    elif action:
        grp = _action_group(action) or (
            _CLEAN_VERBS if any(v in action for v in _CLEAN_VERBS) else None)
        ap = [m for m in pool
              if score_action_match(action, m.get("action")) >= 0.5
              or (grp and any(v in str(m.get("action", "")).lower() for v in grp))]
        if ap:
            pool = ap

    # tally co-present persons, excluding the anchor themselves
    counts: Counter = Counter()
    for m in pool:
        for p in _persons_in(m):
            if is_who_else and p in named_in_q:
                continue
            if rel and p.lower() == rel.lower():
                continue
            counts[p] += 1

    if not counts:                       # only the anchor was present → it was "Me"
        if any((question.get(f"choice_{c}", "") or "").lower().strip() in ME_CHOICE_TOKENS
               for c in "abcd"):
            return map_answer_to_mcq_option_v2("Me", question, min_score=0.3)
        return None

    # set-aware option match (Jaccard between choice names and the tally)
    choices = {c: question.get(f"choice_{c.lower()}", "") for c in ["A", "B", "C", "D"]}
    best_opt, best = None, -1.0
    ev = set(counts)
    for opt, text in choices.items():
        cp = {p for p in PERSON_NAMES | {"Me"} if p.lower() in (text or "").lower()}
        if not cp:
            continue
        sc = len(cp & ev) / len(cp | ev) if (cp | ev) else 0.0
        if sc > best:
            best, best_opt = sc, opt
    return best_opt if best >= 0.25 else None


# ══════════════════════════════════════════════════════════════════════════════
# HabitInsight frequency handler
# ══════════════════════════════════════════════════════════════════════════════

def handle_frequency(
    ranked: List[Dict],
    q_json: Dict,
    question: Dict,
    question_info: Dict,
) -> Optional[str]:
    q_text  = question.get("question", "").lower()
    intent  = q_json.get("query_intent", "what")
    speech_indicators = [
        "not like", "love to", "hate", "prefer", "said",
        "mentioned", "told", "like to eat", "doesn't eat", "don't want",
    ]
    if any(ind in q_text for ind in speech_indicators):
        speech_evidence = [m for m in ranked[:10] if (m.get("speech_content") or "").strip()]
        if speech_evidence:
            return llm_fallback_answer(question, q_json, speech_evidence[:EVIDENCE_K], question_info)
    field = "persons" if intent == "who" else "object"
    e_type = q_json.get("entity", "")
    if e_type in ("tool", "app"):
        field = e_type
    field_counts: Counter = Counter()
    for meta in ranked:
        if field == "persons":
            for p in extract_persons_from_meta(meta):
                field_counts[p] += 1
            if not extract_persons_from_meta(meta):
                field_counts["Me"] += 1
        else:
            val = (meta.get(field) or "").strip()
            if val:
                field_counts[val] += 1
    if not field_counts:
        return llm_fallback_answer(question, q_json, ranked[:EVIDENCE_K], question_info)
    top_val = field_counts.most_common(1)[0][0]
    return map_answer_to_mcq_option_v2(top_val, question, min_score=0.35)


# ══════════════════════════════════════════════════════════════════════════════
# Hierarchy cascade — FIXED (HIER-1, HIER-2, HIER-3)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_persons_from_text(text: str) -> List[str]:
    return [name for name in PERSON_NAMES if name.lower() in text.lower()]

def _slot_digest_to_evidence(slot: Dict, day_label: str) -> Dict:
    """Convert slot digest to evidence dict using ACTUAL slot fields."""
    ao_pairs = slot.get("action_object_pairs", [])
    ao_str   = "; ".join(
        f"{p.get('action','')} {p.get('object','')}".strip()
        for p in ao_pairs[:5]
        if p.get("action") or p.get("object")
    )
    ev = {
        "date":                 day_label,
        "slot":                 slot.get("slot", ""),
        "clip_range":           slot.get("clip_range", []),
        "top_objects":          ", ".join(slot.get("top_objects", [])[:5]),
        "top_actions":          ", ".join(slot.get("top_actions", [])[:5]),
        "action_object_pairs":  ao_str,
        "source":               f"slot_digest_{day_label}_slot{slot.get('slot','')}",
    }
    hi_alpha = slot.get("high_alpha_events", [])
    if hi_alpha:
        ev["high_alpha_events"] = json.dumps(hi_alpha[:3])
    if slot.get("llm_summary"):
        ev["llm_summary"] = slot["llm_summary"]
    return ev

def _daily_digest_to_evidence(daily: Dict, day_label: str) -> Dict:
    """Convert daily digest to evidence dict using ACTUAL daily fields."""
    ev = {
        "date":        day_label,
        "total_fios":  daily.get("total_fios", 0),
        "top_objects": ", ".join(daily.get("top_objects", [])[:8]),
        "top_actions": ", ".join(daily.get("top_actions", [])[:8]),
        "source":      f"daily_digest_{day_label}",
    }
    hi_alpha = daily.get("high_alpha_events", [])
    if hi_alpha:
        ev["high_alpha_events"] = json.dumps(hi_alpha[:3])
    if daily.get("llm_summary"):
        ev["llm_summary"] = daily["llm_summary"]
    return ev

def cascade_to_hierarchy(
    question: Dict,
    q_json: Dict,
    question_info: Dict,
    hier_data: Optional[Dict],
    question_type: str,
    inferred_day: Optional[str] = None,
) -> Optional[str]:
    """
    Cascade to hierarchy when FIO retrieval fails or is insufficient.
    Uses ACTUAL hierarchy_memory.json field names (HIER-1/2/3 fix).

    Slot fields that EXIST:   day, slot, clip_range, fio_count, avg_importance,
                               top_objects, top_actions, action_object_pairs,
                               high_alpha_events, attribute_summary, llm_summary
    Daily fields that EXIST:  day, total_fios, avg_importance, total_clips_processed,
                               total_slots, top_objects, top_actions, peak_slot,
                               high_alpha_events, attribute_summary, slot_breakdown,
                               llm_summary
    Fields that DO NOT EXIST: person_presence_count, speech_highlights,
                               key_events_timeline, speech_of_the_day
    """
    if hier_data is None:
        return None

    query_day    = question_info.get("day", "DAY1")
    target_day   = inferred_day or query_day
    target_day_n = normalize_date(target_day)

    # ── Level 2: Slot digest (HabitInsight) ──────────────────────────────────
    # HIER-1/3 fix: use top_objects, top_actions, action_object_pairs, llm_summary.
    # Person names mined from llm_summary text (person_presence_count absent).
    if question_type == "HabitInsight":
        slots_for_day = {
            k: v for k, v in hier_data.get("slots", {}).items()
            if v.get("day") == target_day_n
        }
        if slots_for_day:
            combined_objects: Counter = Counter()
            combined_actions: Counter = Counter()
            person_mentions:  Counter = Counter()
            slot_evidence = []

            for slot_key, slot in sorted(slots_for_day.items()):
                for obj in slot.get("top_objects", []):
                    combined_objects[obj] += 1
                for act in slot.get("top_actions", []):
                    combined_actions[act] += 1
                for pair in slot.get("action_object_pairs", []):
                    pass  # available if needed for future use

                # Mine person names from llm_summary (HIER-3 fix)
                summary = slot.get("llm_summary", "")
                for name in _extract_persons_from_text(summary):
                    person_mentions[name] += 1
                for ev in slot.get("high_alpha_events", []):
                    for name in _extract_persons_from_text(json.dumps(ev)):
                        person_mentions[name] += 1

                slot_evidence.append(_slot_digest_to_evidence(slot, f"DAY{target_day_n}"))

            intent = q_json.get("query_intent", "what")

            if intent == "who" and person_mentions:
                top = person_mentions.most_common(1)[0][0]
                opt = map_answer_to_mcq_option_v2(top, question, min_score=0.35)
                if opt:
                    return format_mcq_answer(
                        opt, f"Person '{top}' most frequently mentioned in slot summaries."
                    )

            if intent == "what" and combined_objects:
                top_obj = combined_objects.most_common(1)[0][0]
                opt = map_answer_to_mcq_option_v2(top_obj, question, min_score=0.35)
                if opt:
                    return format_mcq_answer(
                        opt, f"Object '{top_obj}' most frequent in slot digests."
                    )

            if slot_evidence:
                return llm_fallback_answer(question, q_json, slot_evidence[:EVIDENCE_K], question_info)

    # ── Level 3: Daily digest ─────────────────────────────────────────────────
    # HIER-2 fix: use top_objects, top_actions, llm_summary, slot_breakdown.
    # key_events_timeline and speech_of_the_day do NOT exist.
    if inferred_day is None:
        all_daily = hier_data.get("daily", {})
        if not all_daily:
            return None
        all_evidence = []
        for day_n_str, day_digest in sorted(all_daily.items(), key=lambda x: int(x[0])):
            day_label = f"DAY{day_n_str}"
            all_evidence.append(_daily_digest_to_evidence(day_digest, day_label))
            # Pull slot summaries via slot_breakdown cross-reference
            for sb in day_digest.get("slot_breakdown", [])[:3]:
                slot_key  = f"{day_n_str}_{sb.get('slot', 0)}"
                full_slot = hier_data.get("slots", {}).get(slot_key)
                if full_slot and full_slot.get("llm_summary"):
                    all_evidence.append({
                        "date":       day_label,
                        "slot":       sb.get("slot"),
                        "llm_summary": full_slot["llm_summary"],
                        "source":     f"slot_summary_{day_label}_s{sb.get('slot',0)}",
                    })
        if all_evidence:
            return llm_fallback_answer(question, q_json, all_evidence[:EVIDENCE_K], question_info)
        return None

    # Single-day lookup — daily keys are str(int): "1", "2", ...
    daily = hier_data.get("daily", {}).get(str(target_day_n))
    if not daily:
        return None

    evidence = [_daily_digest_to_evidence(daily, target_day)]

    # Cross-reference slot_breakdown → full slot entries for richer context
    for sb in daily.get("slot_breakdown", [])[:5]:
        slot_key  = f"{target_day_n}_{sb.get('slot', 0)}"
        full_slot = hier_data.get("slots", {}).get(slot_key)
        if full_slot:
            evidence.append(_slot_digest_to_evidence(full_slot, target_day))

    if evidence:
        return llm_fallback_answer(question, q_json, evidence[:EVIDENCE_K], question_info)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Main QA function
# ══════════════════════════════════════════════════════════════════════════════

def q_a_v2(
    question: Dict[str, Any],
    dual_storage: DualFIOStorage,
    db: str = "opt",
    hier_data: Optional[Dict] = None,
    collection_name: Optional[str] = None,
) -> Optional[str]:
    """
    Complete QA pipeline — final version.

    db="opt" activates OPT-A (alpha_score in ranking) and OPT-B
    (action_sequence expansion for LLM evidence).
    """
    question_type = question.get("type", "")
    q_text        = question.get("question", "")

    # ── Step 1: Extract question metadata ────────────────────────────────────
    query_time    = question.get("query_time") or {}
    question_info = {
        "day":  query_time.get("date", "DAY1"),
        "time": query_time.get("time", 0),
    }
    query_day   = question_info["day"]
    query_day_n = normalize_date(query_day)

    # ── Step 2: Contextual pre-processor ─────────────────────────────────────
    is_ctx, ctx_part, actual_q = detect_contextual(q_text)
    prompt = f"Context: {ctx_part}\nQuestion: {actual_q}" if is_ctx else q_text

    # ── Step 3: Qwen decomposition ────────────────────────────────────────────
    _qa_tokens = {
        "decomp_input": 0, "decomp_output": 0,
        "fallback_input": 0, "fallback_output": 0,
        "n_fios_evidence": 0, "used_llm_fallback": False,
    }
    try:
        _decomp_messages = [
            {"role": "system", "content": Q_decompose.system_message},
            {"role": "user",   "content": prompt},
        ]
        try:
            from Q_decompose import tokenizer as _qa_tokenizer
            _decomp_prompt_str = _qa_tokenizer.apply_chat_template(
                _decomp_messages, tokenize=False, add_generation_prompt=True
            )
            _decomp_inputs = _qa_tokenizer(_decomp_prompt_str, return_tensors="pt")
            _qa_tokens["decomp_input"] = _decomp_inputs["input_ids"].shape[1]
        except Exception:
            _qa_tokens["decomp_input"] = len(prompt) // 4

        response = Q_decompose.call_qwen(
            prompt=prompt, system_message=Q_decompose.system_message
        )
        _qa_tokens["decomp_output"] = len(response) // 4 if response else 0
        print(f"Decomposed: {q_text[:60]} → {response[:300] if response else 'None'} "
              f"[tokens: in={_qa_tokens['decomp_input']} out={_qa_tokens['decomp_output']}]")
        q_json = extract_qwen_json(response)
    except Exception as e:
        print(f"[WARN] Decomposition failed: {e}")
        q_json = {}

    if is_ctx and not q_json.get("context_anchor"):
        q_json["context_anchor"] = ctx_part

    # Carry the natural-language question into retrieval. The decomposed slots
    # for "who was I <verb>ing with" questions collapse to entity="person", so
    # discriminating words ("dancing", "mobile software", "sit next to") are lost
    # from the embed string unless we pass the raw question through.
    q_json["_question_text"] = q_text

    # ── Step 4: Smart retrieval ───────────────────────────────────────────────
    if question_type == "RelationMap":
        fio_results = dual_storage.search_person_filter(
            query=q_json, k=30, query_day=query_day, db=db
        )
    elif question_type == "HabitInsight" and any(
        x in q_text.lower() for x in ["said", "mentioned", "not like", "doesn't", "don't want"]
    ):
        fio_results = dual_storage.search_speech_filter(
            query=q_json, k=30, query_day=query_day, db=db
        )
    else:
        fio_results = dual_storage.search(
            query=q_json, k=30, query_day=query_day, db=db
        )

    raw_metas = (fio_results.get("metadatas") or [[]])[0]
    print(f"Retrieved: {len(raw_metas)} FIOs from {get_collection_name(query_day_n)}")

    # ── Step 5: Temporal filter ───────────────────────────────────────────────
    filtered = filter_by_query_time(raw_metas, question_info)
    print(f"After temporal filter: {len(filtered)} FIOs")

    # ── Step 6: Opt-aware soft ranking (OPT-A) ────────────────────────────────
    # ranked = soft_filter_fio_and_rank(filtered, q_json, db=db)
    # if not ranked:
    #     ranked = filtered
    # print(f"After soft ranking: {len(ranked)} FIOs")
    # This lets temporal questions operate on all retrieved events rather than semantic-score order.
    temp_rel = q_json.get("temporal_relation")

    is_first_last = (
        temp_rel in {"first_time", "last_time"}
        if isinstance(temp_rel, str)
        else any(
            x in {"first_time", "last_time"}
            for x in temp_rel
            if isinstance(x, str)
        )
        if isinstance(temp_rel, (list, tuple, set))
        else False
    )

    try:
        ranked = filtered if is_first_last else soft_filter_fio_and_rank(filtered, q_json, db=db)
        if not ranked:
            ranked = filtered
    except Exception as e:
        print(f"[WARN] ranking failed: {e}")
        ranked = filtered

    # ── OPT-C: Evidence logging ───────────────────────────────────────────────
    _log_evidence(ranked, db=db)

    # ── Step 7: Temporal relation ─────────────────────────────────────────────
    # FIX: restrict to FIOs matching the queried entity/action/person (and the
    # query day for "today" questions) BEFORE taking the first/last extremum, so
    # the answer is the extreme *relevant* event, not the extreme retrieved one.
    temporal = q_json.get("temporal_relation")
    if temporal and temporal not in ("habitual", "none"):
        pool = temporal_candidate_pool(ranked, q_json, q_text, query_day_n)
        ranked_t = apply_temporal_relation(
            pool, temporal, ref_meta=pool[0] if pool else None
        )
        if ranked_t:
            ranked = ranked_t

    # ── Step 8: If empty → hierarchy cascade ─────────────────────────────────
    if not ranked:
        print("FIO retrieval empty → cascading to hierarchy")
        inferred = raw_metas[0].get("date") if raw_metas else None
        hier_answer = cascade_to_hierarchy(
            question, q_json, question_info, hier_data, question_type,
            inferred_day=inferred,
        )
        if hier_answer:
            return hier_answer
        return "Answer: Cannot be determined."

    # ── Step 9: Type-specific handlers ───────────────────────────────────────
    intent = q_json.get("query_intent", "")
    e_type = q_json.get("entity", "")

    # OPT-B helper: expand merged FIO evidence for LLM calls when db="opt"
    def _llm_evidence(fios: List[Dict], max_ev: int = EVIDENCE_K) -> List[Dict]:
        if db == "opt" and _has_merged_fios(fios):
            expanded = expand_opt_evidence(fios, max_evidence=max_ev)
            print(f"  [OPT-B] Expanded {len(fios)} FIOs → {len(expanded)} evidence items")
            return expanded
        return fios[:max_ev]

    # LOGIC-1: hierarchy cascade on deterministic failure for all question types
    def _try_hier_cascade(source_label: str) -> Optional[str]:
        if hier_data is None:
            return None
        inferred = ranked[0].get("date") if ranked else None
        print(f"  [{source_label}] deterministic failed → trying hierarchy cascade")
        return cascade_to_hierarchy(
            question, q_json, question_info, hier_data, question_type,
            inferred_day=inferred,
        )

    # ── EntityLog ─────────────────────────────────────────────────────────────
    if question_type == "EntityLog":

        if intent == "when":
            opt, delta_mins = handle_when(ranked, question, question_info)
            # OPT-B: expand action_sequence in evidence so LLM sees merged steps
            base_ev = _llm_evidence([ranked[0]], max_ev=EVIDENCE_K)
            evidence = [{
                **base_ev[0],
                "_hint_delta_mins":  delta_mins,
                "_hint_delta_human": _mins_to_human(delta_mins),
                "_hint_query_time":  str(question_info.get("time", "")),
                "_hint_fio_time":    str(ranked[0].get("start_time", "")),
            }] + base_ev[1:]
            if opt:
                return format_mcq_answer(opt,
                    f"Time difference: ~{delta_mins} min ({_mins_to_human(delta_mins)}).")
            hier_ans = _try_hier_cascade("EntityLog/when")
            if hier_ans:
                return hier_ans
            return llm_fallback_answer(question, q_json, evidence, question_info)

        if intent == "who":
            person = who_answer(ranked, question, q_json)
            if person:
                opt = map_answer_to_mcq_option_v2(person, question, min_score=0.40)
                if opt:
                    return format_mcq_answer(opt, f"Person evidence: '{person}'.")
                evidence = [{**ranked[0], "_hint_person": person}] + _llm_evidence(ranked[1:], max(EVIDENCE_K - 1, 0))
                return llm_fallback_answer(question, q_json, evidence, question_info)
            hier_ans = _try_hier_cascade("EntityLog/who")
            if hier_ans:
                return hier_ans
            return llm_fallback_answer(question, q_json, _llm_evidence(ranked), question_info)

        if intent == "where":
            loc = resolve_where_answer(ranked[0], temporal)
            opt = map_answer_to_mcq_option_v2(loc, question, min_score=0.40)
            if opt:
                return format_mcq_answer(opt, f"Location evidence: '{loc}'.")
            hier_ans = _try_hier_cascade("EntityLog/where")
            if hier_ans:
                return hier_ans
            evidence = [{**ranked[0], "_hint_location": loc or ""}] + _llm_evidence(ranked[1:], max(EVIDENCE_K - 1, 0))
            return llm_fallback_answer(question, q_json, evidence, question_info)

        if intent == "what":
            field = {"tool": "tool", "app": "app"}.get(e_type, "object")
            val   = ranked[0].get(field, "")
            opt   = map_answer_to_mcq_option_v2(val, question, min_score=0.40)
            if opt:
                return format_mcq_answer(opt, f"{field} evidence: '{val}'.")
            hier_ans = _try_hier_cascade("EntityLog/what")
            if hier_ans:
                return hier_ans
            evidence = [{**ranked[0], "_hint_value": val or ""}] + _llm_evidence(ranked[1:], max(EVIDENCE_K - 1, 0))
            return llm_fallback_answer(question, q_json, evidence, question_info)

    # ── RelationMap ───────────────────────────────────────────────────────────
    if question_type == "RelationMap":
        q_lower = q_text.lower()
        if any(x in q_lower for x in [
            "compared", "missing", "hasn't", "didn't",
            "not participate", "not eat", "not play", "not join"
        ]):
            hier_ans = _try_hier_cascade("RelationMap/set-diff")
            if hier_ans:
                return hier_ans
            return llm_fallback_answer(question, q_json, _llm_evidence(ranked), question_info)

        opt = handle_cooccurrence_v2(ranked, q_json, question)
        if opt:
            return format_mcq_answer(opt, "Co-occurrence conditioned on action/anchor person.")
        opt = handle_cooccurrence(ranked, q_json, question)
        if opt:
            return format_mcq_answer(opt, "Based on co-occurring persons in evidence.")
        hier_ans = _try_hier_cascade("RelationMap/cooccurrence")
        if hier_ans:
            return hier_ans
        return llm_fallback_answer(question, q_json, _llm_evidence(ranked), question_info)

    # ── HabitInsight ──────────────────────────────────────────────────────────
    if question_type == "HabitInsight":
        opt = handle_frequency(ranked, q_json, question, question_info)
        if opt:
            return format_mcq_answer(opt, "Based on frequency/speech evidence.")
        inferred = ranked[0].get("date") if ranked else None
        hier_answer = cascade_to_hierarchy(
            question, q_json, question_info, hier_data, question_type,
            inferred_day=inferred,
        )
        if hier_answer:
            return hier_answer
        return llm_fallback_answer(question, q_json, _llm_evidence(ranked), question_info)

    # ── TaskMaster ────────────────────────────────────────────────────────────
    if question_type == "TaskMaster":
        speech_ranked = [m for m in ranked if m.get("speech_content", "").strip()]
        raw_evidence  = speech_ranked[:EVIDENCE_K] if speech_ranked else ranked[:EVIDENCE_K]
        evidence      = _llm_evidence(raw_evidence)

        val = ranked[0].get("object") or ranked[0].get("action", "")
        opt = map_answer_to_mcq_option_v2(val, question, min_score=0.35)
        if opt:
            return format_mcq_answer(opt, f"Plan evidence: '{val}'.")

        # LOGIC-2: cascade when speech evidence is thin
        if len(speech_ranked) < 2:
            hier_ans = _try_hier_cascade("TaskMaster")
            if hier_ans:
                return hier_ans
        return llm_fallback_answer(question, q_json, evidence, question_info)

    # ── EventRecall ───────────────────────────────────────────────────────────
    # if question_type == "EventRecall":
    #     # LOGIC-1: try hierarchy first for multi-step context
    #     hier_ans = _try_hier_cascade("EventRecall")
    #     if hier_ans:
    #         return hier_ans
    #     return llm_fallback_answer(question, q_json, _llm_evidence(ranked), question_info)
    # ── EventRecall ───────────────────────────────────────────────────────────
    if question_type == "EventRecall":

        # deterministic time resolution for "when was the last/first time ..."
        if intent == "when":
            opt, delta_mins = handle_when(ranked, question, question_info)
            if opt:
                return format_mcq_answer(
                    opt,
                    f"Time difference: ~{delta_mins} min ({_mins_to_human(delta_mins)}).")
            base_ev  = _llm_evidence(ranked, max_ev=EVIDENCE_K)
            evidence = [{
                **base_ev[0],
                "_hint_delta_mins":  delta_mins,
                "_hint_delta_human": _mins_to_human(delta_mins),
                "_hint_query_time":  str(question_info.get("time", "")),
                "_hint_fio_time":    str(ranked[0].get("start_time", "")),
            }] + base_ev[1:]
            return llm_fallback_answer(question, q_json, evidence, question_info)

        # deterministic "what activity / what did we <verb>" resolution.
        # The answer is the object/action of the relevant event, not an LLM guess.
        if intent == "what":
            cand_fios: List[Dict] = []
            grp = _action_group(q_json.get("action"))
            if grp is not None:                    # e.g. "what did we BUY" → the buy event
                gmatch = [m for m in filtered if _action_group(m.get("action")) is grp]
                gmatch.sort(key=_meta_time_key,
                            reverse=(q_json.get("temporal_relation") != "first_time"))
                cand_fios += gmatch[:1]
            cand_fios.append(ranked[0])
            for tgt in cand_fios:
                for val in (tgt.get("object", ""),
                            tgt.get("action", ""),
                            f"{tgt.get('action','')} {tgt.get('object','')}".strip()):
                    if not val.strip():
                        continue
                    opt = map_answer_to_mcq_option_v2(val, question, min_score=0.32)
                    if opt:
                        return format_mcq_answer(
                            opt, f"Activity/object evidence: '{val.strip()}'.")
            return llm_fallback_answer(question, q_json, _llm_evidence(ranked), question_info)

        # deterministic "who" resolution (set-aware, action/anchor conditioned)
        if intent == "who":
            opt = handle_cooccurrence_v2(ranked, q_json, question)
            if not opt:
                person = who_answer(ranked, question, q_json)
                opt = map_answer_to_mcq_option_v2(person, question, min_score=0.40)
            if opt:
                return format_mcq_answer(opt, "Co-occurrence / person evidence.")
            return llm_fallback_answer(question, q_json, _llm_evidence(ranked), question_info)

        # non-"when"/"what"/"who" EventRecall (e.g. "why") → LLM with top-5 evidence
        return llm_fallback_answer(question, q_json, _llm_evidence(ranked), question_info)

    # ── Generic fallback ──────────────────────────────────────────────────────
    if intent == "who":
        person = who_answer(ranked, question, q_json)
        opt = map_answer_to_mcq_option_v2(person, question, min_score=0.40)
        if opt:
            return format_mcq_answer(opt, f"Person evidence: '{person}'.")
        return llm_fallback_answer(question, q_json,
            [{**ranked[0], "_hint_person": person or ""}] + _llm_evidence(ranked[1:], max(EVIDENCE_K - 1, 0)),
            question_info)

    if intent == "what":
        field = {"tool": "tool", "app": "app"}.get(e_type, "object")
        val = ranked[0].get(field, "")
        opt = map_answer_to_mcq_option_v2(val, question, min_score=0.40)
        if opt:
            return format_mcq_answer(opt, f"{field} evidence: '{val}'.")
        return llm_fallback_answer(question, q_json,
            [{**ranked[0], "_hint_value": val or ""}] + _llm_evidence(ranked[1:], max(EVIDENCE_K - 1, 0)),
            question_info)

    if intent == "where":
        loc = resolve_where_answer(ranked[0], temporal)
        opt = map_answer_to_mcq_option_v2(loc, question, min_score=0.40)
        if opt:
            return format_mcq_answer(opt, f"Location: '{loc}'.")
        return llm_fallback_answer(question, q_json,
            [{**ranked[0], "_hint_location": loc or ""}] + _llm_evidence(ranked[1:], max(EVIDENCE_K - 1, 0)),
            question_info)

    if intent == "when":
        opt, delta_mins = handle_when(ranked, question, question_info)
        base_ev  = _llm_evidence([ranked[0]], max_ev=EVIDENCE_K)
        evidence = [{
            **base_ev[0],
            "_hint_delta_mins":  delta_mins,
            "_hint_delta_human": _mins_to_human(delta_mins),
        }] + base_ev[1:]
        if opt:
            return format_mcq_answer(opt,
                f"Time difference: ~{delta_mins} min ({_mins_to_human(delta_mins)}).")
        return llm_fallback_answer(question, q_json, evidence, question_info)

    print(f"[FALLBACK] No handler matched: type={question_type} intent={intent}")
    return llm_fallback_answer(question, q_json, _llm_evidence(ranked), question_info)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def get_valid_days(max_day: int) -> List[str]:
    return [f"DAY{i}" for i in range(1, max_day + 1)]

def filter_questions_by_day(data, valid_days, q_type, camera_wearer=None):
    questions, answers = [], []
    for item in data:
        if item.get("type") != q_type:
            #print("type mismatch", item.get("type"), q_type)
            continue
        #q_day = item.get("query_time", {}).get("date") # for jake QA
        q_day = item.get("query_time", {}).get("date")
        if q_day in valid_days:
            questions.append(item)
            answers.append(item.get("answer"))
    return questions, answers


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("ENTERED MAIN")
    camera_wearer="Jake" #"Tasha" # or "Jake"
    print(f"Camera wearer set to: {camera_wearer}")
    set_camera_wearer(camera_wearer)

    job_name = os.environ.get("SLURM_JOB_NAME")
    job_id   = os.environ.get("SLURM_JOB_ID")
    # exp_name = (
    #     f"{job_name}_{job_id}" if job_name and job_id
    #     else f"qa_v5_final_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # )

    # --- build config -> retrieval-ranker ablation (single source of truth) ---
    # Rule: keep the importance term in ranking ONLY when the build produced a
    # real importance_score. Drop it when importance is removed or degenerate.
    _IMP_PRESENT = {"full", "no_s3"}     # importance signal is real -> use it
    _IMP_ABSENT  = {"no_s2", "no_s1", "s3_only"}    # keep-all / uniform / no proto stats

    def ranker_ablation_for(build_config: str) -> str:
        bc = build_config.strip().lower()
        if bc in _IMP_PRESENT:
            return "full"     # (0.53, 0.29, 0.18)
        if bc in _IMP_ABSENT:
            return "no_imp"   # (0.65, 0.35, 0.00)  -- see note below
        raise ValueError(f"Unknown build config {build_config!r}; "
                        "add it to _IMP_PRESENT or _IMP_ABSENT.")

    BUILD_CONFIG = os.environ.get("BUILD_CONFIG", "full")
    MEM_WINDOW   = os.environ.get("MEM_WINDOW",   "day1-7")                     # set this per run
    MAX_DAY    = int(os.environ.get("MAX_DAY", "7")) # set it for per new mwm abl run
    #ABLATION   = "full"  # "full" | "no_e" | "no_a" | "no_imp" | "equal" | "only_e"
    ABLATION     = ranker_ablation_for(BUILD_CONFIG)
    exp_name = f"{BUILD_CONFIG}_{MEM_WINDOW}_K{EVIDENCE_K}"
    print(f"Experiment Name: {exp_name}")
    logger = start_experiment(exp_name)
    print(f"Experiment logger initialized: {logger}")

    #DATA_PATH= os.environ.get("DATA_PATH") full no_s1 no_s2 no_s3 s3_only
    #ab_folder_name = f"memory_build_v2_7days/fio_cache_{MEM_WINDOW}_mem/{BUILD_CONFIG}"
    ab_folder_name="BUILD_7day_ablation_full__delta_h_threshold=0.7"#"BUILD_7day_ablation_full__context_history_weight=0.3"
    print(ab_folder_name)
    # ab_folder_name = os.environ.get("AB_FOLDER_NAME",
    #     "memory_build_v2_7days/fio_cache_day1_mem/s3_only") #BUILD_7day_ablation_full__lambda_decay=2.6742e-6
    # #ab_folder_name="BUILD_7day_tau_full__tau_threshold=0.3"#tasha
    #ab_folder_name="BUILD_7dAy_0406" # for raw
    #ab_folder_name="BUILD_7day_ablation_full__lambda_decay=1.6045e-6"#"BUILD_7day_ablation_no_s2" # BUILD_7day_ablation_no_s3 #BUILD_7day_ablation_s3_only #BUILD_7day_ablation_no_s1
    #DATA_PATH  = "/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoLifeQA_A3_TASHA.json" #tasha
    DATA_PATH  = "/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoLifeQA_A1_JAKE.json" #jake
    #DATA_PATH="/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoGPT/split_temporal/EgoLifeQA_A1_JAKE_part4_gt24h.json" 
    HIER_PATH  = f"/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/{ab_folder_name}/build_report.json"
    RAW_DB_DIR = f"/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/{ab_folder_name}/chroma_fio_raw"
    #RAW_DB_DIR = f"/tmp/24CS92R01/BUILD_local_full/full/chroma_fio_raw"
    OPT_DB_DIR = f"/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/{ab_folder_name}/chroma_fio_opt"
    #OPT_DB_DIR="/tmp/24CS92R01/BUILD_local_full/full/chroma_fio_opt"
    #DB_MODE    = "opt"   # "opt" = full system | "raw" = ablation baseline
    DB_MODE    = os.environ.get("DB_MODE", "opt")
    
    

    QUESTION_TYPES = [
        "EntityLog", "EventRecall", "RelationMap", "HabitInsight", "TaskMaster"
    ]
    # QUESTION_TYPES = [
    #      "TaskMaster"
    # ]
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    USE_HIERARCHY = False #True
    hier_data = None
    if USE_HIERARCHY:
        if os.path.exists(HIER_PATH):
            with open(HIER_PATH, "r") as f:
                hier_data = json.load(f)
            print(f"[INFO] Hierarchy loaded: {HIER_PATH}")
        else:
            print(f"[WARN] No hierarchy at {HIER_PATH} — running without hierarchy")
    else:
        print("[INFO] USE_HIERARCHY=False — ablation mode (no hierarchy cascade)")

    # ── Stage opt DB to node-local disk for safe, concurrent-friendly reads ──
    from nfs_db_staging import StagedChromaDB
    _stager      = StagedChromaDB(local_root=os.environ.get("LOCAL_DB_ROOT"),
                                  sync_on_exit=False)   # read-only: nothing to sync back
    OPT_DB_LOCAL = _stager.stage_in("chroma_fio_opt", OPT_DB_DIR if DB_MODE == "opt" else None)
    RAW_DB_LOCAL = _stager.stage_in("chroma_fio_raw", RAW_DB_DIR if DB_MODE == "raw" else None)

    dual_storage = DualFIOStorage(
        raw_db_dir=RAW_DB_LOCAL or RAW_DB_DIR,
        opt_db_dir=OPT_DB_LOCAL or OPT_DB_DIR,
        store_raw=(DB_MODE == "raw"),
        store_opt=(DB_MODE == "opt"),
    )

    # # ── SANITY: what does the eval's own opt client see on this path? ──
    # print(f"[SANITY] opt_db.db_dir = {dual_storage.opt_db.db_dir}")
    # _probe = dual_storage.opt_db.client.get_or_create_collection(
    #     name="collection_day1_to_day7",
    #     embedding_function=dual_storage.opt_db.embedding_function,
    # )
    # print(f"[SANITY] collection_day1_to_day7 count = {_probe.count()}")

    all_results  = {}
    global_correct  = 0
    global_total    = 0
    global_answered = 0
    global_grounded = 0

    for d in [MAX_DAY]:
        valid_days = get_valid_days(d)
        day_key    = f"DAY{d}"
        all_results[day_key] = {}

        print(f"\n{'='*80}")
        print(f"Experiment: DAY1 to DAY{d} | DB: {DB_MODE}")
        print(f"{'='*80}")

        for q_type in QUESTION_TYPES:
            print(f"\nQuestion Type: {q_type}")
            questions, correct_answers = filter_questions_by_day(data, valid_days, q_type,camera_wearer=camera_wearer)
            if not questions:
                print("No questions found")
                continue

            predict_answers = []

            for idx, q in enumerate(questions):
                print(f"\n{idx}: {q['question']} (Day: {q['query_time']['date']})")
                answer = q_a_v2(
                    question=q,
                    dual_storage=dual_storage,
                    db=DB_MODE,
                    hier_data=hier_data,
                )
                pred = extract_single_option_answer(answer) if answer else None
                print(f"Model Answer: {answer}")
                print(f"Extracted Option: {pred}")
                ans_v = correct_answers[idx]
                print(f"Correct Answer: {ans_v} - {q.get(f'choice_{ans_v.lower()}','')}")
                is_grounded = (
                    pred is not None and
                    answer is not None and
                    "cannot be determined" not in (answer or "").lower()
                )
                predict_answers.append((pred, is_grounded))
                print(f"  [Summary Q{idx}] type={q.get('type','?')} "
                      f"pred={pred} correct={ans_v} "
                      f"{'✓' if pred==ans_v else '✗'} "
                      f"{'[grounded]' if is_grounded else '[cascade]'}")

            correct_count  = 0
            answered_count = 0
            for (pred, is_grounded), corr in zip(predict_answers, correct_answers):
                global_total += 1
                if pred is None:
                    continue
                answered_count  += 1
                global_answered += 1
                if pred == corr:
                    correct_count  += 1
                    global_correct += 1
                    if is_grounded:
                        global_grounded += 1

            accuracy         = correct_count / answered_count if answered_count else 0.0
            overall_accuracy = correct_count / len(correct_answers) if correct_answers else 0.0
            all_results[day_key][q_type] = {
                "accuracy":         accuracy,
                "overall_accuracy": overall_accuracy,
                "correct":          correct_count,
                "total_answered":   answered_count,
                "total_questions":  len(correct_answers),
            }
            print(f"\nAccuracy for {q_type}: {accuracy*100:.2f}% "
                  f"({correct_count}/{answered_count}), "
                  f"Total Q: {len(correct_answers)} for {day_key}")

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    for day, res in all_results.items():
        print(f"\n{day}")
        for q_type, metrics in res.items():
            print(f"  {q_type:15s}: "
                  f"{metrics['overall_accuracy']*100:.2f}% overall "
                  f"({metrics['correct']}/{metrics['total_questions']}) | "
                  f"eval acc={metrics['accuracy']*100:.2f}% "
                  f"({metrics['correct']}/{metrics['total_answered']})")

    N   = max(global_total,   1)
    A   = max(global_answered, 1)
    acc          = global_correct  / N * 100
    ans_rate     = global_answered / N * 100
    acc_at_ans   = global_correct  / A * 100
    grounded_acc = global_grounded / N * 100

    print(f"\n{'─'*80}")
    print(f"{'Method':<35} {'Acc.':>6} {'Ans. Rate':>10} {'Acc@Ans.':>10} {'Grounded Acc.':>14}")
    print(f"{'─'*80}")
    print(f"{'EgoRAG Baseline (GPT)':<35} {'36.0':>6} {'100.0':>10} {'36.0':>10} {'–':>14}")
    print(f"{'Reproduced Baseline (Qwen)':<35} {'31.6':>6} {'99.6':>10} {'31.7':>10} {'31.6':>14}")
    print(f"{'Hierarchical FIO Memory (Qwen)':<35} {acc:>6.1f} {ans_rate:>10.1f} "
          f"{acc_at_ans:>10.1f} {grounded_acc:>14.1f}")
    print(f"{'─'*80}")
    print(f"Gap vs EgoRAG baseline: {acc - 36.0:+.1f}pp")
    print(f"{'─'*80}")

    logger.info(f"q_a_v6_final complete. Global: {global_correct}/{global_total}")