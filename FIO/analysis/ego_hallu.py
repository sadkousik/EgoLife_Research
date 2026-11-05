#!/usr/bin/env python3
"""
Hallucination analysis for EgoRAG (baseline, Jake, 7 days, no DeL).

Per-question classification:
    Grounded                  evidence_card non-empty AND model_option committed
    Self-admitted hallucination  evidence_card empty AND model_answer contains an
                              admission phrase (e.g. "None of the provided video
                              events mention…")
    Silent parametric         evidence_card empty AND no admission phrase AND
                              model_option committed (i.e. just a bare letter)
    Refusal                   model_option is None  (excluded from Hall numerator)

Per-type reporting: N, Accuracy, ER, GA, Hall%, Self%, Silent%, H∧Correct.

ER  = grounded / N
GA  = (correct AND grounded) / correct
Hall% = (self_admitted + silent) / N
"""

import json
import re
from collections import defaultdict
from pathlib import Path
exp_name="20260618_094817_DB_EGO_life_TASHA_1706_EgoLifeQA_A3_TASHA_results"#answer_results
IN_JSON  = Path(f"/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/answer_results/{exp_name}.json")
OUT_JSON = Path(f"./tasha_e/egorag_hallucination_analysis_7d_{exp_name}.json")
OUT_MD   = Path(f"./tasha_e/egorag_hallucination_report_7d_{exp_name}.md")


# ──────────────────────────────────────────────────────────────────────────────
# Self-admission phrases — same family as the FIO analysis, plus EgoRAG-specific
# phrasings observed in the data ("None of the provided video events mention…")
# ──────────────────────────────────────────────────────────────────────────────
SELF_ADMIT_PATTERNS = [
    r"none of the (?:provided|given) (?:video )?(?:events|related events)",
    r"do(?:es)? not (?:mention|specify|indicate|provide|state|confirm|contain)",
    r"does not (?:explicitly )?(?:mention|specify|indicate|provide|state|confirm)",
    r"no (?:direct )?(?:evidence|mention|reference|information)",
    r"is not (?:explicitly )?(?:mentioned|specified|indicated|provided|possible)",
    r"based on (?:typical|common|context|the (?:given )?information|the context)",
    r"most plausible",
    r"cannot be (?:determined|confirmed|inferred)",
    r"insufficient (?:evidence|information)",
    r"by (?:elimination|process of elimination)",
    r"makes the most sense",
    r"is the (?:best|most likely) guess",
    r"no valid answer",
    r"not possible to determine",
]
SELF_ADMIT_RE = re.compile("|".join(SELF_ADMIT_PATTERNS), re.IGNORECASE)


def classify(r):
    """Returns a dict of classification flags for one EgoRAG result row."""
    pred         = r.get("model_option")
    committed    = pred is not None and pred != "None"
    has_evidence = bool(r.get("evidence_card"))
    model_answer = r.get("model_answer") or ""

    out = {
        "is_grounded":        False,
        "is_hallucinated":    False,
        "hallucination_kind": None,    # "self_admitted" | "silent_parametric"
        "is_refusal":         False,
        "is_self_admitted":   False,
    }

    if not committed:
        out["is_refusal"] = True
        return out

    if has_evidence:
        out["is_grounded"] = True
        return out

    # committed, no evidence retrieved → hallucinated; split by admission
    out["is_hallucinated"]  = True
    out["is_self_admitted"] = bool(SELF_ADMIT_RE.search(model_answer))
    out["hallucination_kind"] = (
        "self_admitted" if out["is_self_admitted"] else "silent_parametric"
    )
    return out


def aggregate(results):
    buckets = defaultdict(list)
    for r in results:
        qt = r.get("metadata", {}).get("type", "Unknown")
        buckets[qt].append(r)
    buckets["__OVERALL__"] = results

    rows = {}
    for qt, rs in buckets.items():
        N = len(rs)
        n_correct           = sum(1 for r in rs if r.get("is_correct"))
        n_grounded          = sum(1 for r in rs if r["_cls"]["is_grounded"])
        n_grounded_correct  = sum(
            1 for r in rs
            if r.get("is_correct") and r["_cls"]["is_grounded"]
        )
        n_halluc            = sum(1 for r in rs if r["_cls"]["is_hallucinated"])
        n_halluc_self       = sum(
            1 for r in rs
            if r["_cls"]["hallucination_kind"] == "self_admitted"
        )
        n_halluc_silent     = sum(
            1 for r in rs
            if r["_cls"]["hallucination_kind"] == "silent_parametric"
        )
        n_halluc_correct    = sum(
            1 for r in rs
            if r.get("is_correct") and r["_cls"]["is_hallucinated"]
        )
        n_refusal           = sum(1 for r in rs if r["_cls"]["is_refusal"])

        rows[qt] = {
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
            "n_refusal":          n_refusal,
        }
    return rows


def pct(x):
    return f"{x*100:5.2f}%"


def print_table(rows):
    order = ["EntityLog", "EventRecall", "RelationMap", "HabitInsight",
             "TaskMaster", "__OVERALL__"]
    print()
    print(f"{'Type':<14} {'N':>4} {'Acc':>7} {'ER':>7} {'GA':>7} "
          f"{'Hall%':>7} {'Self%':>7} {'Silent%':>8} {'H∧Corr':>7} {'Refuse':>6}")
    print("-" * 92)
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
              f"{m['halluc_correct']:>7} "
              f"{m['n_refusal']:>6}")
    print("-" * 92)
    print("Hall%   = (self_admitted + silent_parametric) / N")
    print("Self%   = self-admitted hallucinations / N  (model_answer admits no evidence)")
    print("Silent% = silent parametric (committed answer w/ empty evidence_card, no admission) / N")
    print("H∧Corr  = answers that are BOTH hallucinated AND correct (lucky guesses)")
    print("Refuse  = model_option is None (excluded from Hall numerator)")


def write_markdown(rows, path):
    order = ["EntityLog", "EventRecall", "RelationMap", "HabitInsight",
             "TaskMaster", "__OVERALL__"]
    lines = []
    lines.append("# Hallucination analysis — EgoRAG (Jake, 7 days, baseline, no DeL)\n")
    lines.append("## Definitions\n")
    lines.append(
        "- **Accuracy** = `# correct / # total`\n"
        "- **ER** = `# grounded / # total`. Grounded ⇔ `evidence_card` is "
        "non-empty AND the model committed to an option. "
        "*(For EgoRAG, `model_answer` is usually just the letter, so we use "
        "the presence of `evidence_card` as the strongest available "
        "grounding signal — same proxy as in the EgoRAG paper.)*\n"
        "- **GA** = `# (correct AND grounded) / # correct`\n"
        "- **Hallucination rate** = `(self_admitted + silent_parametric) / N`. "
        "Both buckets require `evidence_card` empty AND `model_option` "
        "committed — i.e. the LLM picked an option without any retrieved "
        "memory.\n"
        "  - **Self-admitted**: `model_answer` contains an admission phrase "
        "such as *\"None of the provided video events mention…\"* or *\"not "
        "possible to determine\"*.\n"
        "  - **Silently parametric**: no admission — `model_answer` is "
        "typically just the option letter.\n"
        "- **Refusal**: `model_option = None`. Not counted in Hall numerator.\n"
    )
    lines.append("## Per question-type\n")
    lines.append("| Type | N | Acc | ER | GA | Hall% | Self% | Silent% | H∧Correct | Refusal |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
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
            f"{m['halluc_correct']} | {m['n_refusal']} |"
        )
    lines.append("")
    lines.append("## Methodology notes\n")
    lines.append(
        "- EgoRAG's `model_answer` field is typically a single letter "
        "(median length ≈ 11 chars including the `assistant\\n` prefix), so "
        "a textual overlap check between the model's explanation and the "
        "retrieved `evidence_card` snippets is not meaningful for most "
        "rows. The presence-of-evidence proxy is the strongest signal "
        "available.\n"
        "- The ~20 rows where `model_answer` is longer than ~20 chars are "
        "all empty-evidence rows where the LLM verbalises a refusal "
        "(e.g. *\"None of the provided video events mention any purchases…\"*) "
        "before being forced to commit an option anyway. These are "
        "captured by the Self% column.\n"
        "- Side-by-side comparison with FIO requires noting that EgoRAG "
        "has no deterministic-handler path (no path-B equivalent). Every "
        "EgoRAG answer goes through the LLM, so the ER ceiling is set by "
        "the retrieval recall of `extract_evidence(...)`.\n"
    )
    path.write_text("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = json.loads(IN_JSON.read_text())
    results = data["results"]
    print(f"[load] {len(results)} records from {IN_JSON.name}")
    print(f"       reported accuracy = {data.get('accuracy', 0):.4f}")

    for r in results:
        r["_cls"] = classify(r)

    rows = aggregate(results)
    print_table(rows)

    OUT_JSON.write_text(json.dumps(
        {"per_type": rows, "records": results},
        indent=2, ensure_ascii=False
    ))
    write_markdown(rows, OUT_MD)
    print(f"\n[saved] {OUT_JSON}")
    print(f"[saved] {OUT_MD}")