#!/usr/bin/env python3
"""
Grounding analysis for the new-format log (no [Evidence] block printed).

This is the analyser behind the paper's GA numbers; fio/grounding.py copies its
rules so scripts/qa.py can compute GA directly. Input: a QA log path set at the top.

Rule (option 1 — self-admission only, STRICT):
  - Path B (deterministic handler)  → grounded by construction
  - Path A / C (LLM-authored answer) → grounded iff the explanation contains
                                       NONE of:
                                         (A) explicit no-evidence admissions
                                             ("evidence does not mention…",
                                              "cannot be determined", …)
                                         (B) hedging verbs ("likely",
                                             "appears to", "suggests",
                                             "might be", "implies", …)
                                         (C) choice-elimination language
                                             ("among the given options",
                                              "matches option B",
                                              "the only female option", …)
  - Cascade (pred=None / Answer: null) → refusal (not in any numerator)

Reports per question-type: N, Accuracy, ER, GA, GC, Hall%, Self%, H∧Correct.

Path detection:
  - Path B  : `Model Answer: Answer: X` followed by deterministic template
              explanation, and no `[LLM fallback] evidence=N FIOs` line.
  - Path A  : `[LLM fallback] evidence=N FIOs` line present.
  - Path C  : `[FALLBACK] No handler matched: …` line present (scored like A).
  - Cascade : `Extracted Option: None` or `Answer: null`.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
# EDIT THESE
# ────────────────────────────────────────────────────────────────────────────
exp_name="ans_fio_without_meta_21520(nodel)"
FIO_LOG = Path(f"/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/log_best/{exp_name}.log")

LOG_PATH = FIO_LOG           # input log
OUT_JSON = Path(f"./abl/_{exp_name}_grounding_analysis-1.json")     # per-record + per-type stats
OUT_MD   = Path(f"./abl/_{exp_name}_grounding_report-1.md")         # human-readable report
#LOG_PATH = Path("./your_log_file.log")           # input log
#OUT_JSON = Path("./grounding_analysis.json")     # per-record + per-type stats
#OUT_MD   = Path("./grounding_report.md")         # human-readable report
REPORT_NAME = "grounding_selfadmit_strict"       # title in the markdown


# ────────────────────────────────────────────────────────────────────────────
# Self-admission patterns (STRICTER VERSION).
# Three blocks:
#   (A) Explicit no-evidence admissions (the original loose-rule set).
#   (B) Hedging verbs — model speculating rather than reading evidence.
#   (C) Choice-elimination language — model picks from priors / option set.
# Any match = self-admitted hallucination on path A / C.
#
# Block (D) ("common", "typical usage") is intentionally NOT added — those
# words are lexically common and would over-flag genuinely-grounded answers.
# ────────────────────────────────────────────────────────────────────────────

# (A) Explicit no-evidence admissions
_EXPLICIT_ADMIT = [
    r"evidence does not",
    r"does not (?:explicitly )?(?:mention|specify|indicate|provide|state|confirm|directly|contain|show)",
    r"do(?:es)? not (?:directly )?show",
    r"no (?:direct )?(?:evidence|mention|reference|information|context|indication|clear)",
    r"is not (?:explicitly )?(?:mentioned|specified|indicated|provided)",
    r"based on (?:typical|common|context|the context|the given|the available)",
    r"most plausible",
    r"is the only (?:choice|option) (?:that |involving|mentioned)",
    r"aligns with typical usage",
    r"cannot be (?:determined|confirmed|definitively)",
    r"insufficient (?:evidence|information)",
    r"by (?:elimination|process of elimination)",
    r"makes the most sense",
    r"is the (?:best|most likely) guess",
    r"assuming",
    r"not possible to determine",
    r"limited (?:evidence|information|context)",
    r"without (?:additional|further) context",
    r"none of the (?:provided|given|other|actions|choices)",
    r"could be any of",
    r"weak evidence",
    r"no specific (?:mention|information)",
    r"plausible (?:answer|choice|option) (?:based on|given)",
    r"most (?:likely|plausible) (?:answer|choice|option|based)",
    r"there is no (?:clear|direct|specific)",
    r"there's no (?:clear|direct|specific)",
]

# (B) Hedging verbs — the model is speculating, not retrieving
_HEDGE_VERBS = [
    r"\blikely\b",
    r"\bmight (?:be|have|imply|suggest)",
    r"\bmay (?:be|have|imply|suggest)",
    r"\bcould (?:be|have|imply|suggest|indicate)",
    r"\bperhaps\b",
    r"\bpossibly\b",
    r"\bprobably\b",
    r"\bappears?\s+to\b",
    r"\bseems?\s+to\b",
    r"\bseems?\s+like\b",
    r"\bimplies?\b",
    r"\bimplying\b",
    r"\bsuggests?\b",
    r"\bsuggesting\b",
    r"\binfer(?:red|s|ring)?\b",
]

# (C) Choice-elimination / option-distribution reasoning
_CHOICE_ELIM = [
    r"among the (?:given |provided |available )?(?:choices|options)",
    r"given the (?:choices|options|available choices)",
    r"closest (?:to|match|and safest)",
    r"matches (?:with|option|the description)",
    r"aligns (?:with|most closely with) (?:option|choice)",
    r"the only (?:option|choice|name|female|male|new)",
    r"out of the (?:given |provided )?(?:choices|options)",
    r"of the (?:given |provided )?(?:choices|options)",
]

SELF_ADMIT_PATTERNS = _EXPLICIT_ADMIT + _HEDGE_VERBS + _CHOICE_ELIM
SELF_ADMIT_RE = re.compile("|".join(SELF_ADMIT_PATTERNS), re.IGNORECASE)


# ────────────────────────────────────────────────────────────────────────────
# Line patterns
# ────────────────────────────────────────────────────────────────────────────
Q_START_RE   = re.compile(r"^(?P<idx>\d+):\s+(?P<q>.+?)\s+\(Day:\s+(?P<day>DAY\d+)\)\s*$")
LLM_EV_RE    = re.compile(r"\[LLM fallback\]\s+evidence=(?P<n>\d+)\s+FIOs")
NO_HANDLER_RE = re.compile(r"^\[FALLBACK\]\s+No handler matched")
MODEL_ANS_RE = re.compile(r"^Model Answer:\s*(?P<rest>.*)$")
EXTRACTED_RE = re.compile(r"^Extracted Option:\s*(?P<opt>\S+)\s*$")
CORRECT_RE   = re.compile(r"^Correct Answer:\s*(?P<opt>[A-Z])\s*-\s*(?P<text>.*?)\s*$")
SUMMARY_RE   = re.compile(
    r"\[Token summary Q(?P<idx>\d+)\]\s+type=(?P<qtype>\S+)\s+"
    r"pred=(?P<pred>\S+)\s+correct=(?P<corr>\S+)\s+(?P<mark>[✓✗])"
)
EXPLAIN_RE   = re.compile(
    r"Explain your choice in one sentence[:.]?\s*(?P<rest>.*)$"
)
TEMPLATE_RE  = re.compile(
    r"^Explain your choice in one sentence\.\s*"
    r"(?:Person evidence|Plan evidence|Time difference|object evidence|Based on)"
)


# ────────────────────────────────────────────────────────────────────────────
# Parser
# ────────────────────────────────────────────────────────────────────────────
def parse_log(path: Path) -> list:
    records = []
    cur = None
    explanation_lines = []
    state = "idle"   # "idle" | "after_model_ans"
    saw_llm_fallback = False
    saw_no_handler   = False
    saw_template     = False

    def flush(rec):
        if rec is None:
            return
        if explanation_lines:
            rec["explanation"] = " ".join(
                line.strip() for line in explanation_lines if line.strip()
            ).strip()
        records.append(rec)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")

            m = Q_START_RE.match(line)
            if m:
                # close previous
                if cur is not None:
                    cur["saw_llm_fallback"] = saw_llm_fallback
                    cur["saw_no_handler"]   = saw_no_handler
                    cur["saw_template"]     = saw_template
                    flush(cur)
                cur = {
                    "q_idx":        int(m.group("idx")),
                    "day":          m.group("day"),
                    "question":     m.group("q").strip(),
                    "qtype":        None,
                    "gt_option":    None,
                    "gt_text":      None,
                    "pred_option":  None,
                    "correct":      None,
                    "explanation":  None,
                    "llm_evidence_n": None,
                    "path":         None,
                }
                explanation_lines = []
                state = "idle"
                saw_llm_fallback = False
                saw_no_handler   = False
                saw_template     = False
                continue

            if cur is None:
                continue

            m = LLM_EV_RE.search(line)
            if m:
                cur["llm_evidence_n"] = int(m.group("n"))
                saw_llm_fallback = True
                continue

            if NO_HANDLER_RE.search(line):
                saw_no_handler = True
                continue

            m = MODEL_ANS_RE.match(line)
            if m:
                rest = m.group("rest")
                m2 = EXPLAIN_RE.search(rest)
                if m2 and m2.group("rest"):
                    explanation_lines.append(m2.group("rest"))
                state = "after_model_ans"
                continue

            # detect the deterministic-template line shape
            if TEMPLATE_RE.match(line):
                saw_template = True
                # also capture as explanation
                m2 = EXPLAIN_RE.search(line)
                if m2 and m2.group("rest"):
                    explanation_lines.append(m2.group("rest"))
                state = "after_model_ans"
                continue

            m = EXPLAIN_RE.search(line)
            if m:
                if m.group("rest"):
                    explanation_lines.append(m.group("rest"))
                state = "after_model_ans"
                continue

            m = EXTRACTED_RE.match(line)
            if m:
                opt = m.group("opt")
                cur["pred_option"] = None if opt == "None" else opt
                state = "idle"
                continue

            m = CORRECT_RE.match(line)
            if m:
                cur["gt_option"] = m.group("opt")
                cur["gt_text"]   = m.group("text").strip()
                continue

            m = SUMMARY_RE.search(line)
            if m:
                cur["qtype"]   = m.group("qtype")
                cur["correct"] = (m.group("mark") == "✓")
                if cur["pred_option"] is None:
                    p = m.group("pred")
                    cur["pred_option"] = None if p in ("None", "null") else p

                # path inference
                if cur["pred_option"] is None:
                    cur["path"] = "cascade"
                elif saw_no_handler:
                    cur["path"] = "C"
                elif saw_llm_fallback:
                    cur["path"] = "A"
                else:
                    cur["path"] = "B"

                cur["saw_llm_fallback"] = saw_llm_fallback
                cur["saw_no_handler"]   = saw_no_handler
                cur["saw_template"]     = saw_template
                flush(cur)

                cur = None
                explanation_lines = []
                state = "idle"
                saw_llm_fallback = False
                saw_no_handler   = False
                saw_template     = False
                continue

            # explanation continuation in the after-model-ans zone
            if state == "after_model_ans":
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("[") or stripped.startswith("Using ") \
                        or stripped.startswith("Answer:"):
                    continue
                explanation_lines.append(stripped)

    if cur is not None:
        cur["saw_llm_fallback"] = saw_llm_fallback
        cur["saw_no_handler"]   = saw_no_handler
        cur["saw_template"]     = saw_template
        flush(cur)

    # keep only well-formed records
    records = [r for r in records if r.get("qtype") is not None]
    return records


# ────────────────────────────────────────────────────────────────────────────
# Classification
# ────────────────────────────────────────────────────────────────────────────
def classify(rec):
    path = rec.get("path")
    explanation = rec.get("explanation") or ""
    pred = rec.get("pred_option")
    committed = pred is not None and pred != "None"

    out = {
        "is_self_admitted":   False,
        "is_grounded":        False,
        "is_hallucinated":    False,
        "hallucination_kind": None,
    }

    if path == "cascade":
        return out

    if path == "B":
        out["is_grounded"] = committed
        return out

    if not committed:
        return out

    out["is_self_admitted"] = bool(SELF_ADMIT_RE.search(explanation))
    if not out["is_self_admitted"]:
        out["is_grounded"] = True
    else:
        out["is_hallucinated"]    = True
        out["hallucination_kind"] = "self_admitted"
    return out


# ────────────────────────────────────────────────────────────────────────────
# Aggregate
# ────────────────────────────────────────────────────────────────────────────
def aggregate(records):
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
        n_halluc_correct = sum(
            1 for r in rs
            if r["correct"] and r["_cls"]["is_hallucinated"]
        )
        n_cascade = sum(1 for r in rs if r["path"] == "cascade")
        n_A = sum(1 for r in rs if r["path"] == "A")
        n_B = sum(1 for r in rs if r["path"] == "B")
        n_C = sum(1 for r in rs if r["path"] == "C")

        rows[qtype] = {
            "N":                  N,
            "n_correct":          n_correct,
            "accuracy":           n_correct / N if N else 0.0,
            "ER":                 n_grounded / N if N else 0.0,
            "GA":                 n_grounded_correct / n_correct if n_correct else 0.0,
            "hallucination_rate": n_halluc / N if N else 0.0,
            "halluc_self_rate":   n_halluc_self / N if N else 0.0,
            "halluc_correct":     n_halluc_correct,
            "n_grounded":         n_grounded,
            "n_grounded_correct": n_grounded_correct,
            "n_hallucinated":     n_halluc,
            "n_halluc_self":      n_halluc_self,
            "n_cascade":          n_cascade,
            "paths":              {"A": n_A, "B": n_B, "C": n_C, "cascade": n_cascade},
        }
    return rows


# ────────────────────────────────────────────────────────────────────────────
# Reporting
# ────────────────────────────────────────────────────────────────────────────
def pct(x):
    return f"{x*100:5.2f}%"


def print_table(rows):
    order = ["EntityLog", "EventRecall", "RelationMap", "HabitInsight",
             "TaskMaster", "__OVERALL__"]
    print()
    print(f"{'Type':<14} {'N':>4} {'Acc':>7} {'ER':>7} {'GA':>7} "
          f"{'GC':>4} {'Hall%':>7} {'Self%':>7} {'H∧Corr':>7}")
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
              f"{m['n_grounded_correct']:>4} "
              f"{pct(m['hallucination_rate']):>7} "
              f"{pct(m['halluc_self_rate']):>7} "
              f"{m['halluc_correct']:>7}")
    print("-" * 80)
    print("Rule: grounded ⇔ path B OR (path A/C with no self-admission phrase)")
    print("GC     = # (correct AND grounded)")
    print("Hall%  = self-admitted hallucinations / N")
    print("H∧Corr = answers that are BOTH correct AND hallucinated (lucky guess)")


def write_markdown(rows, path, report_name):
    order = ["EntityLog", "EventRecall", "RelationMap", "HabitInsight",
             "TaskMaster", "__OVERALL__"]
    lines = []
    lines.append(f"# {report_name}\n")
    lines.append("## Definitions (STRICT self-admission rule)\n")
    lines.append(
        "- **Accuracy** = `# correct / # total`\n"
        "- **ER** = `# grounded / # total`. Grounded:\n"
        "  - Path B (deterministic handler) → grounded by construction.\n"
        "  - Path A / C (LLM fallback) → grounded iff the explanation contains "
        "**none** of:\n"
        "    - **(A) Explicit no-evidence admissions** — *\"evidence does not "
        "mention…\"*, *\"cannot be determined\"*, *\"insufficient evidence\"*, "
        "etc.\n"
        "    - **(B) Hedging verbs** — *\"likely\"*, *\"appears to\"*, "
        "*\"seems to\"*, *\"suggests\"*, *\"might be\"*, *\"could be\"*, "
        "*\"implies\"*, *\"infer\"*, etc. — i.e. the model is speculating "
        "rather than reading evidence.\n"
        "    - **(C) Choice-elimination language** — *\"among the given "
        "options\"*, *\"matches option B\"*, *\"the only female option\"*, "
        "*\"closest match\"*, etc. — i.e. the model picked from the option "
        "distribution rather than from retrieval.\n"
        "- **GA** = `# (correct AND grounded) / # correct`\n"
        "- **GC** = `# (correct AND grounded)` (raw count)\n"
        "- **Hall%** = `# hallucinated / # total` "
        "(all hallucinations are self-admitted under this rule).\n"
        "- **Self%** = same as Hall% under this rule (kept for table parity).\n"
        "- **H∧Correct** = # correct answers that are also hallucinated "
        "(model hedged or eliminated choices, but happened to pick the right "
        "letter).\n"
        "- Cascade rows (`pred=None` / `Answer: null`) are refusals; counted "
        "in N but not in any hallucination numerator.\n"
        "- *Note: textual overlap with retrieved FIOs is not used here because "
        "this log does not print the FIO fields. The rule is purely textual: "
        "hedging or choice-elimination prose triggers the hallucination flag.*\n"
    )
    lines.append("## Per question-type\n")
    lines.append("| Type | N | Acc | ER | GA | GC | Hall% | Self% | H∧Correct |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for qt in order:
        if qt not in rows:
            continue
        m = rows[qt]
        label = "**OVERALL**" if qt == "__OVERALL__" else qt
        lines.append(
            f"| {label} | {m['N']} | {pct(m['accuracy'])} | "
            f"{pct(m['ER'])} | {pct(m['GA'])} | {m['n_grounded_correct']} | "
            f"{pct(m['hallucination_rate'])} | "
            f"{pct(m['halluc_self_rate'])} | "
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
    lines.append("## Notes\n")
    lines.append(
        "- This log lacks the `[Evidence] top-N FIOs` block, so the textual "
        "overlap component of the earlier ER/GA analysis cannot run on it. "
        "The numbers here are upper-bounds on grounding — any case where "
        "the LLM committed an option with confident-sounding (but actually "
        "unsupported) prose escapes detection.\n"
        "- Path B (deterministic handler) is auto-grounded by construction: "
        "the handler reads a FIO field directly into the answer template.\n"
        "- Path C is the no-handler fallback. It uses the same LLM machinery "
        "as Path A and is scored with the same self-admission rule.\n"
    )
    path.write_text("\n".join(lines))


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    records = parse_log(LOG_PATH)
    print(f"[parse] {len(records)} questions from {LOG_PATH.name}")

    for r in records:
        r["_cls"] = classify(r)

    rows = aggregate(records)
    print_table(rows)

    OUT_JSON.write_text(json.dumps(
        {"per_type": rows, "records": records},
        indent=2, ensure_ascii=False
    ))
    write_markdown(rows, OUT_MD, REPORT_NAME)
    print(f"\n[saved] {OUT_JSON}")
    print(f"[saved] {OUT_MD}")