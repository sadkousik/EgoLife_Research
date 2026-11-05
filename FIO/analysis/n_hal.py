#!/usr/bin/env python3
"""
STRICT hallucination analysis over multiple fio_records JSON files.

Difference vs the loose version:
    A path-A / path-C answer is grounded only if the explanation overlaps with
    >= STRICT_MIN_MATCHES distinct FIO-field tokens (was 1 in the loose rule).

Looks for every file in INPUT_DIR whose name starts with "fio" and ends with
".json", runs the strict classifier on each, writes per-run sidecar JSON +
Markdown report, and finally prints/saves a combined 5×6 GC comparison table.

Edit the three paths near the top to fit your machine.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
# EDIT THESE
# ────────────────────────────────────────────────────────────────────────────
INPUT_DIR     = Path("/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/G_result/delh")              # folder containing fio*.json
OUTPUT_DIR    = Path("/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/G_result/delh/strict_out_n")   # where reports get written
INPUT_GLOB    = "fio*.json"                        # pattern: filename starts with fio
STRICT_MIN_MATCHES = 2                             # tighten the overlap rule

# ────────────────────────────────────────────────────────────────────────────
CAMERA_WEARER = "Jake"
ME_TOKENS = {"me", "i", "my", "mine", "myself"}

# SELF_ADMIT_PATTERNS = [
#     r"evidence does not",
#     r"does not (?:explicitly )?(?:mention|specify|indicate|provide|state|confirm)",
#     r"no (?:direct )?(?:evidence|mention|reference|information)",
#     r"is not (?:explicitly )?(?:mentioned|specified|indicated|provided)",
#     r"based on (?:typical|common|context|the context)",
#     r"most plausible",
#     r"is the only (?:choice|option) (?:that |involving|mentioned)",
#     r"aligns with typical usage",
#     r"cannot be (?:determined|confirmed)",
#     r"insufficient (?:evidence|information)",
#     r"by (?:elimination|process of elimination)",
#     r"makes the most sense",
#     r"is the (?:best|most likely) guess",
#     r"assuming",
# ]
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

    # ── added: hedging / soft-guessing ─────────────────────────────────
    r"\bpresumably\b",
    r"\bpresumed\b",
    r"reasonable to (?:assume|infer|conclude)",
    r"\bi (?:think|believe|infer|assume|suspect|guess)\b",
    r"not (?:directly |explicitly )?(?:supported|stated|confirmed|specified)",
    r"without (?:explicit|specific|direct) (?:evidence|mention|information)",
    r"common sense",
    r"could (?:be|have been)",
    r"might (?:be|have been)",
    r"best (?:fit|guess|match)",
    r"only logical (?:choice|option|answer)",
    r"\binfer(?:red|ence)?\b",
    r"\bimpl(?:y|ies|ied)\b",

    # ── aggressive single-word hedges (comment out if over-flagging) ────
    r"\bappears?\b",
    r"\bseems?\b",
    r"\bsuggests?\b",
    r"\blikely\b",
    r"\bprobably\b",
    r"\btypically\b",
    r"\busually\b",
    r"\bgenerally\b",
]
SELF_ADMIT_RE = re.compile("|".join(SELF_ADMIT_PATTERNS), re.IGNORECASE)

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


def fio_terms(fio):
    out = []
    if fio.get("person"):
        persons = {p.strip().lower()
                   for p in re.split(r"[,/]", fio["person"]) if p.strip()}
        persons.discard("—")
        out.append(("person", persons))
    for kind in ("action", "object", "loc"):
        if fio.get(kind):
            terms = tokenize(fio[kind])
            if terms:
                out.append((kind, terms))
    if fio.get("speech"):
        sp = re.sub(r"^[\w ]+ said:\s*", "", fio["speech"])
        terms = tokenize(sp)
        if terms:
            out.append(("speech", terms))
    return out


def collect_overlap(explanation, fios, camera_wearer=CAMERA_WEARER):
    """Return list of (kind, token) overlap hits, deduplicated by (kind, token)."""
    if not explanation or not fios:
        return []

    exp_tokens = tokenize(explanation)
    raw_exp_low = explanation.lower()
    has_me_ref = bool(ME_TOKENS & exp_tokens) or any(
        re.search(rf"\b{re.escape(tok)}\b", raw_exp_low) for tok in ("i", "me")
    )

    matched = set()
    cam_low = camera_wearer.lower()
    for fio in fios:
        for kind, terms in fio_terms(fio):
            if kind == "person" and cam_low in terms and has_me_ref:
                matched.add((kind, "<me>"))
            for h in (terms & exp_tokens):
                matched.add((kind, h))
    return sorted(matched)


def classify(rec):
    path = rec.get("path")
    explanation = rec.get("explanation") or ""
    pred = rec.get("pred_option")
    committed = pred is not None and pred != "None"

    out = {
        "is_self_admitted":   False,
        "n_overlap":          0,
        "matched_terms":      [],
        "is_grounded":        False,
        "is_hallucinated":    False,
        "hallucination_kind": None,
    }

    if path == "cascade":
        return out

    if path == "B":
        out["is_grounded"]   = committed
        out["n_overlap"]     = STRICT_MIN_MATCHES  # path B auto-passes the strict rule
        return out

    if not committed:
        return out

    out["is_self_admitted"] = bool(SELF_ADMIT_RE.search(explanation))
    matched = collect_overlap(explanation, rec.get("fios_top_k") or [])
    out["matched_terms"] = matched
    out["n_overlap"]     = len(matched)

    has_strict_overlap = len(matched) >= STRICT_MIN_MATCHES
    if has_strict_overlap and not out["is_self_admitted"]:
        out["is_grounded"] = True
    else:
        out["is_hallucinated"] = True
        out["hallucination_kind"] = (
            "self_admitted" if out["is_self_admitted"] else "silent_parametric"
        )
    return out


def aggregate(records):
    buckets = defaultdict(list)
    for r in records:
        buckets[r["qtype"]].append(r)
    buckets["__OVERALL__"] = records

    rows = {}
    for qt, rs in buckets.items():
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
        rows[qt] = {
            "N": N,
            "n_correct": n_correct,
            "accuracy":  n_correct / N if N else 0.0,
            "ER":        n_grounded / N if N else 0.0,
            "GA":        n_grounded_correct / n_correct if n_correct else 0.0,
            "hallucination_rate": n_halluc / N if N else 0.0,
            "halluc_self_rate":   n_halluc_self / N if N else 0.0,
            "halluc_silent_rate": n_halluc_silent / N if N else 0.0,
            "halluc_correct":     n_halluc_correct,
            "n_grounded":         n_grounded,
            "n_grounded_correct": n_grounded_correct,
            "n_hallucinated":     n_halluc,
            "n_halluc_self":      n_halluc_self,
            "n_halluc_silent":    n_halluc_silent,
        }
    return rows


def pct(x):
    return f"{x*100:5.2f}%"


def write_markdown_for_run(stem, rows, path):
    order = ["EntityLog", "EventRecall", "RelationMap", "HabitInsight",
             "TaskMaster", "__OVERALL__"]
    lines = []
    lines.append(f"# STRICT hallucination analysis — {stem}\n")
    lines.append(
        f"**STRICT rule:** path-A / path-C answers count as grounded only when "
        f"the explanation overlaps with **≥ {STRICT_MIN_MATCHES} distinct "
        f"FIO-field tokens** (loose rule was ≥ 1). Path B (deterministic) "
        f"remains auto-grounded.\n"
    )
    lines.append("## Per question-type\n")
    lines.append("| Type | N | Acc | ER | GA | GC | Hall% | Self% | Silent% | H∧Correct |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
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
            f"{pct(m['halluc_silent_rate'])} | "
            f"{m['halluc_correct']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def write_comparison_table(per_run, path):
    types = ["EntityLog", "EventRecall", "RelationMap", "HabitInsight",
             "TaskMaster", "__OVERALL__"]
    lines = []
    lines.append(f"# STRICT comparison — grounded-correct counts (≥{STRICT_MIN_MATCHES} overlap)\n")

    # GC counts
    lines.append("## Grounded-correct counts\n")
    header = "| Run | " + " | ".join(
        "Overall" if t == "__OVERALL__" else t for t in types[-1:] + types[:-1]
    ) + " |"
    sep = "|---|" + "|".join(["---:"] * len(types)) + "|"
    lines.append(header)
    lines.append(sep)
    ordered_types = types[-1:] + types[:-1]   # Overall first
    for stem, rows in per_run.items():
        cells = [str(rows[t]["n_grounded_correct"]) for t in ordered_types]
        lines.append(f"| {stem} | " + " | ".join(cells) + " |")
    lines.append("")

    # GC rates
    lines.append("## Grounded-correct rate (GC / N)\n")
    lines.append(header)
    lines.append(sep)
    for stem, rows in per_run.items():
        cells = [pct(rows[t]["n_grounded_correct"] / rows[t]["N"])
                 for t in ordered_types]
        lines.append(f"| {stem} | " + " | ".join(cells) + " |")
    lines.append("")

    # Acc / ER / GA / Hall%
    lines.append("## Overall metrics per run\n")
    lines.append("| Run | N | Acc | ER | GA | GC | Hall% | H∧Correct |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for stem, rows in per_run.items():
        m = rows["__OVERALL__"]
        lines.append(
            f"| {stem} | {m['N']} | {pct(m['accuracy'])} | "
            f"{pct(m['ER'])} | {pct(m['GA'])} | {m['n_grounded_correct']} | "
            f"{pct(m['hallucination_rate'])} | {m['halluc_correct']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def print_comparison(per_run):
    types = ["__OVERALL__", "EntityLog", "EventRecall", "RelationMap",
             "HabitInsight", "TaskMaster"]
    print(f"\n{'='*100}")
    print(f"STRICT comparison — grounded-correct counts (≥{STRICT_MIN_MATCHES} overlap)")
    print(f"{'='*100}")
    print(f"  {'Run':<50}" + "".join(
        f"{'Overall' if t == '__OVERALL__' else t:>14}" for t in types
    ))
    for stem, rows in per_run.items():
        line = f"  {stem:<50}"
        for t in types:
            line += f"{rows[t]['n_grounded_correct']:>14}"
        print(line)

    print(f"\n{'Run':<50}{'N':>5}{'Acc':>8}{'ER':>8}{'GA':>8}{'GC':>5}{'Hall%':>8}{'H∧C':>5}")
    print("-" * 100)
    for stem, rows in per_run.items():
        m = rows["__OVERALL__"]
        print(f"{stem:<50}{m['N']:>5}{pct(m['accuracy']):>8}"
              f"{pct(m['ER']):>8}{pct(m['GA']):>8}"
              f"{m['n_grounded_correct']:>5}"
              f"{pct(m['hallucination_rate']):>8}"
              f"{m['halluc_correct']:>5}")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(p for p in INPUT_DIR.glob(INPUT_GLOB) if p.is_file())
    if not input_files:
        raise SystemExit(f"No files matching {INPUT_GLOB} in {INPUT_DIR}")
    print(f"[found] {len(input_files)} input files in {INPUT_DIR}:")
    for p in input_files:
        print(f"   - {p.name}")

    per_run = {}
    for input_path in input_files:
        stem = input_path.stem
        print(f"\n[run] {stem}")
        records = json.loads(input_path.read_text())
        for r in records:
            r["_cls"] = classify(r)
        rows = aggregate(records)
        per_run[stem] = rows

        # per-run outputs
        out_json = OUTPUT_DIR / f"hallucination_analysis_strict__{stem}.json"
        out_md   = OUTPUT_DIR / f"hallucination_report_strict__{stem}.md"
        out_json.write_text(json.dumps(
            {"per_type": rows, "records": records},
            indent=2, ensure_ascii=False
        ))
        write_markdown_for_run(stem, rows, out_md)
        ov = rows["__OVERALL__"]
        print(f"   N={ov['N']} Acc={pct(ov['accuracy'])} "
              f"GC={ov['n_grounded_correct']} "
              f"Hall%={pct(ov['hallucination_rate'])} "
              f"H∧Correct={ov['halluc_correct']}")
        print(f"   [saved] {out_json.name}")
        print(f"   [saved] {out_md.name}")

    # combined comparison
    print_comparison(per_run)
    comp_path = OUTPUT_DIR / "strict_comparison_table.md"
    write_comparison_table(per_run, comp_path)
    print(f"\n[saved] {comp_path}")