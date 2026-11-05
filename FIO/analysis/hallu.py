#!/usr/bin/env python3
"""
Hallucination analysis over fio_records.json.

Per question (path A or C only — i.e. LLM-authored answers):
  - is_self_admitted : explanation contains a phrase admitting no evidence
                       ("evidence does not", "based on context", etc.)
  - has_fio_overlap  : explanation references any (person | object | action)
                       from fios_top_k (loose: ≥1 overlap is enough)
                       with Jake ↔ {me, I, my, mine, myself} treated as a match.
  - is_hallucinated  : pred is not None AND NOT(has_fio_overlap) (i.e.
                       committed an answer but did not ground it).
  - is_grounded      : pred is not None AND has_fio_overlap AND NOT(is_self_admitted)

Path B (deterministic handler): grounded by construction (reads FIO field).
Path cascade: refusal — pred=None, neither grounded nor hallucinated.

Per question type we report:
    N, Accuracy, ER, GA, Hallucination rate (split into self-admitted /
    silently-parametric).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

dir="del"
exp_name="BUILD_7day_ablation_full__history_window=5_680"
IN_JSON  = Path("/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/G_result/hist/fio_records_BUILD_7day_ablation_full__history_window=5_680.json")
OUT_JSON = Path(f"./{dir}/hallucination_analysis_{exp_name}.json")
OUT_MD   = Path(f"./{dir}/hallucination_report_{exp_name}.md")

CAMERA_WEARER = "Jake"
ME_TOKENS = {"me", "i", "my", "mine", "myself"}


# name report
REPORT_NAME = exp_name #  need to change for markdown title

# ──────────────────────────────────────────────────────────────────────────────
# Phrases that explicitly admit "I don't have the evidence for this"
# ──────────────────────────────────────────────────────────────────────────────
SELF_ADMIT_PATTERNS = [
    r"evidence does not",
    r"does not (?:explicitly )?(?:mention|specify|indicate|provide|state|confirm)",
    r"no (?:direct )?(?:evidence|mention|reference|information)",
    r"is not (?:explicitly )?(?:mentioned|specified|indicated|provided)",
    r"based on (?:typical|common|context|the context)",
    r"most plausible",
    r"is the only (?:choice|option) (?:that |involving|mentioned)",
    r"aligns with typical usage",
    r"cannot be (?:determined|confirmed)",
    r"insufficient (?:evidence|information)",
    r"by (?:elimination|process of elimination)",
    r"makes the most sense",
    r"is the (?:best|most likely) guess",
    r"assuming",
]
SELF_ADMIT_RE = re.compile("|".join(SELF_ADMIT_PATTERNS), re.IGNORECASE)

# Words we don't want to count as overlap signal
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with",
    "from", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "as", "if",
    "then", "than", "so", "such", "no", "not", "yes", "do", "does", "did",
    "have", "has", "had", "will", "would", "could", "should", "may", "might",
    "can", "one", "two", "first", "last", "some", "any", "all", "more",
    "most", "many", "few", "they", "them", "their", "he", "she", "his", "her",
    "him", "we", "us", "our", "you", "your", "thing", "things", "something",
    "someone", "people", "person", "evidence", "answer", "choice", "option",
    "based", "context", "sentence", "explain",
}


def tokenize(s):
    if not s:
        return set()
    toks = re.findall(r"[A-Za-z]+", s.lower())
    return {t for t in toks if t not in STOPWORDS and len(t) > 1}


def fio_terms(fio, camera_wearer=CAMERA_WEARER):
    """All the surface terms a FIO contributes to overlap matching.
    Returns a list of (kind, terms_set) pairs.
    """
    out = []
    # person — split on commas, lowercase
    if fio.get("person"):
        persons = {p.strip().lower() for p in re.split(r"[,/]", fio["person"]) if p.strip()}
        persons.discard("—")
        out.append(("person", persons))
    # object / action — content words
    for kind in ("action", "object", "loc"):
        if fio.get(kind):
            terms = tokenize(fio[kind])
            if terms:
                out.append((kind, terms))
    # speech — the substantive words inside the quote
    if fio.get("speech"):
        # drop the "X said:" preamble if present
        sp = re.sub(r"^[\w ]+ said:\s*", "", fio["speech"])
        terms = tokenize(sp)
        if terms:
            out.append(("speech", terms))
    return out


def check_overlap(explanation, fios, camera_wearer=CAMERA_WEARER):
    """LOOSE: any single (person | action | object | loc | speech) term hit
    counts as overlap. Camera-wearer name in FIOs matches me-tokens in
    explanation.
    Returns: (has_overlap: bool, matched_terms: list[(kind, term)])
    """
    if not explanation or not fios:
        return False, []

    exp_tokens = tokenize(explanation)
    raw_exp_low = explanation.lower()

    # detect first-person reference once
    has_me_ref = bool(ME_TOKENS & exp_tokens) or any(
        re.search(rf"\b{re.escape(tok)}\b", raw_exp_low) for tok in ("i", "me")
    )

    matched = []
    cam_low = camera_wearer.lower()

    for fio in fios:
        for kind, terms in fio_terms(fio, camera_wearer):
            # camera-wearer ↔ me-tokens match
            if kind == "person" and cam_low in terms and has_me_ref:
                matched.append((kind, "<me>"))
            hits = terms & exp_tokens
            for h in hits:
                matched.append((kind, h))

    return (len(matched) > 0), matched


# ──────────────────────────────────────────────────────────────────────────────
# Classify each record
# ──────────────────────────────────────────────────────────────────────────────
def classify(rec):
    """Returns dict with classification flags added."""
    path = rec.get("path")
    explanation = rec.get("explanation") or ""
    pred = rec.get("pred_option")
    committed = pred is not None and pred != "None"

    out = {
        "is_self_admitted":   False,
        "has_fio_overlap":    False,
        "matched_terms":      [],
        "is_grounded":        False,
        "is_hallucinated":    False,
        "hallucination_kind": None,    # "self_admitted" | "silent_parametric" | None
    }

    if path == "cascade":
        # refusal — nothing committed
        return out

    if path == "B":
        # deterministic handler — grounded by construction
        out["is_grounded"]   = committed
        out["has_fio_overlap"] = True  # the handler literally wrote a FIO field
        return out

    # path A or C: LLM authored the explanation
    if not committed:
        return out

    out["is_self_admitted"] = bool(SELF_ADMIT_RE.search(explanation))
    has_overlap, matched = check_overlap(explanation, rec.get("fios_top_k") or [])
    out["has_fio_overlap"] = has_overlap
    out["matched_terms"]   = matched

    if has_overlap and not out["is_self_admitted"]:
        out["is_grounded"] = True
    else:
        out["is_hallucinated"] = True
        out["hallucination_kind"] = (
            "self_admitted" if out["is_self_admitted"] else "silent_parametric"
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate
# ──────────────────────────────────────────────────────────────────────────────
def aggregate(records):
    """Compute per-type and overall stats."""
    buckets = defaultdict(list)
    for r in records:
        buckets[r["qtype"]].append(r)
    buckets["__OVERALL__"] = records

    rows = {}
    for qtype, rs in buckets.items():
        N = len(rs)
        n_correct = sum(1 for r in rs if r["correct"])
        n_grounded = sum(1 for r in rs if r["_cls"]["is_grounded"])
        n_grounded_correct = sum(
            1 for r in rs
            if r["correct"] and r["_cls"]["is_grounded"]
        )
        n_halluc = sum(1 for r in rs if r["_cls"]["is_hallucinated"])
        n_halluc_self = sum(
            1 for r in rs
            if r["_cls"]["hallucination_kind"] == "self_admitted"
        )
        n_halluc_silent = sum(
            1 for r in rs
            if r["_cls"]["hallucination_kind"] == "silent_parametric"
        )
        n_halluc_correct = sum(
            1 for r in rs
            if r["correct"] and r["_cls"]["is_hallucinated"]
        )
        n_cascade = sum(1 for r in rs if r["path"] == "cascade")
        n_pathA   = sum(1 for r in rs if r["path"] == "A")
        n_pathB   = sum(1 for r in rs if r["path"] == "B")
        n_pathC   = sum(1 for r in rs if r["path"] == "C")

        rows[qtype] = {
            "N":                  N,
            "n_correct":          n_correct,
            "accuracy":           n_correct / N if N else 0.0,
            "ER":                 n_grounded / N if N else 0.0,
            "GA":                 n_grounded_correct / n_correct if n_correct else 0.0,
            "hallucination_rate": n_halluc / N if N else 0.0,
            "halluc_self_rate":   n_halluc_self / N if N else 0.0,
            "halluc_silent_rate": n_halluc_silent / N if N else 0.0,
            "halluc_correct":     n_halluc_correct,
            "n_grounded":         n_grounded,
            "n_grounded_correct": n_grounded_correct,
            "n_hallucinated":     n_halluc,
            "n_halluc_self":      n_halluc_self,
            "n_halluc_silent":    n_halluc_silent,
            "n_cascade":          n_cascade,
            "paths":              {"A": n_pathA, "B": n_pathB, "C": n_pathC,
                                   "cascade": n_cascade},
        }
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────
def pct(x):
    return f"{x*100:5.2f}%"


def print_table(rows):
    order = ["EntityLog", "EventRecall", "RelationMap", "HabitInsight",
             "TaskMaster", "__OVERALL__"]
    print()
    print(f"{'Type':<14} {'N':>4} {'Acc':>7} {'ER':>7} {'GA':>7} "
          f"{'Hall%':>7} {'Self%':>7} {'Silent%':>8} {'H∧Corr':>7}")
    print("-" * 80)
    for qt in order:
        if qt not in rows:
            continue
        m = rows[qt]
        label = "OVERALL" if qt == "__OVERALL__" else qt
        print(f"{label:<14} {m['N']:>4} "
              f"{pct(m['accuracy']):>7} "
              f"{pct(m['ER']):>7} "
              f"{pct(m['GA']):>7} "
              f"{pct(m['hallucination_rate']):>7} "
              f"{pct(m['halluc_self_rate']):>7} "
              f"{pct(m['halluc_silent_rate']):>8} "
              f"{m['halluc_correct']:>7}")
    print("-" * 80)
    print("Hall%   = hallucination rate (silent_parametric + self_admitted) / N")
    print("Self%   = self-admitted hallucinations / N")
    print("Silent% = silently parametric (committed answer, no FIO overlap, no admission) / N")
    print("H∧Corr  = answers that are BOTH hallucinated AND correct (lucky guesses)")


def write_markdown(rows, path):
    order = ["EntityLog", "EventRecall", "RelationMap", "HabitInsight",
             "TaskMaster", "__OVERALL__"]
    lines = []
    lines.append(f"# {REPORT_NAME}\n")
    lines.append("## Definitions\n")
    lines.append(
        "- **Accuracy** = `# correct / # total`\n"
        "- **ER** = `# grounded / # total` — grounded = the answer references "
        "(person | object | action | loc | speech) from a retrieved FIO and "
        "the explanation does not self-admit lack of evidence. Path B "
        "(deterministic handler) is grounded by construction.\n"
        "- **GA** = `# (correct AND grounded) / # correct`\n"
        "- **Hallucination rate** = `# hallucinated / # total`, hallucinated = "
        "model committed an option but explanation either self-admits no "
        "evidence (e.g. *\"evidence does not mention…\"*) or has zero overlap "
        "with retrieved FIO fields.\n"
        "  - **Self-admitted** = explanation contains a phrase admitting "
        "the evidence is missing.\n"
        "  - **Silently parametric** = no admission, but no FIO content "
        "appears in the explanation either — answer came from priors.\n"
        "- Cascade rows (`pred=None`) are refusals; counted in N but not in "
        "the numerator of any hallucination metric.\n"
    )
    lines.append("## Per question-type\n")
    lines.append("| Type | N | Acc | ER | GA | Hall% | Self% | Silent% | H∧Correct |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for qt in order:
        if qt not in rows:
            continue
        m = rows[qt]
        label = "**OVERALL**" if qt == "__OVERALL__" else qt
        lines.append(
            f"| {label} | {m['N']} | {pct(m['accuracy'])} | "
            f"{pct(m['ER'])} | {pct(m['GA'])} | "
            f"{pct(m['hallucination_rate'])} | "
            f"{pct(m['halluc_self_rate'])} | "
            f"{pct(m['halluc_silent_rate'])} | "
            f"{m['halluc_correct']} |"
        )
    lines.append("")
    lines.append("## Path breakdown per type\n")
    lines.append("| Type | A (LLM) | B (deterministic) | C (no-handler) | cascade |")
    lines.append("|---|---:|---:|---:|---:|")
    for qt in order:
        if qt not in rows:
            continue
        m = rows[qt]
        label = "**OVERALL**" if qt == "__OVERALL__" else qt
        p = m["paths"]
        lines.append(
            f"| {label} | {p['A']} | {p['B']} | {p['C']} | {p['cascade']} |"
        )
    lines.append("")
    lines.append("## Notes on the methodology\n")
    lines.append(
        "- Overlap is *loose*: a single matching content word counts. Tightening "
        "to ≥2 matches would shrink ER and grow the hallucination rate; the "
        "loose setting gives a conservative (lower-bound) hallucination estimate.\n"
        "- Camera-wearer name in FIOs is treated as matching first-person tokens "
        "(`i`, `me`, `my`, `mine`, `myself`) in the explanation, matching the "
        "`set_camera_wearer('Jake')` config.\n"
        "- Path B (deterministic) is auto-grounded — the handler reads the "
        "answer straight out of the top-ranked FIO field.\n"
        "- Path C (no-handler fallback) uses the same machinery as A and is "
        "scored with the same overlap rule.\n"
        "- `H∧Correct` is the count of correct answers that are also "
        "hallucinated (the model guessed the right letter without grounding). "
        "These inflate accuracy but are caught by GA going down.\n"
    )
    path.write_text("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    records = json.loads(IN_JSON.read_text())
    print(f"[load] {len(records)} records from {IN_JSON}")

    for r in records:
        r["_cls"] = classify(r)

    rows = aggregate(records)
    print_table(rows)

    # save classified records + per-type stats
    OUT_JSON.write_text(json.dumps(
        {"per_type": rows, "records": records},
        indent=2, ensure_ascii=False
    ))
    write_markdown(rows, OUT_MD)
    print(f"\n[saved] {OUT_JSON}")
    print(f"[saved] {OUT_MD}")