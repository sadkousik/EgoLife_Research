#!/usr/bin/env python3
"""
Extract one structured record per question from the FIO ablation log.

Output: a single JSON file (list of dicts) with fields:
    q_idx, qtype, day, question,
    gt_option, gt_text, pred_option, correct,
    fios_retrieved, fios_top_k (list of parsed FIO dicts),
    llm_evidence_n, path (A | B | C | cascade),
    explanation, summary_tag

    A — LLM fallback path. Retrieval succeeded, the top-k FIOs were stuffed into a Qwen prompt, and the LLM wrote both the option and the free-form explanation.
    B — Deterministic handler path. A type-specific handler (e.g. EntityLog/who) read the answer directly out of the top-ranked FIO's fields and emitted a fixed-template explanation like Person evidence: 'Me'. — no LLM call.
    C — No-handler fallback path. No deterministic handler matched the question's (type, intent) signature, so the system logged [FALLBACK] No handler matched and routed to the LLM with retrieved evidence (effectively a path-A answer reached via a different door).
    cascade — Ungrounded/refusal path. The model returned None or "cannot be determined", so the summary line tagged it [cascade] and no committed answer was produced.
"""

import json
import re
from pathlib import Path
#EgoRAG/log_upto_18_06/ctx0.3_873(34.6).log
#/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/log/eval_abl_A6_lam5d_tau020_cpu_24263.log
#..../eval_abl_A7_lam14d_tau020_cpu_24264.log
exp_name="deltaH0.2_u_26680"#"ans_nos3_25985"#"ans_nos1_537"
FIO_LOG = Path(f"/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/log/tempo/{exp_name}.log")
OUT     = Path(f"./delh/fio_records_{exp_name}.json")


# ──────────────────────────────────────────────────────────────────────────────
# Regexes for the line types we care about
# ──────────────────────────────────────────────────────────────────────────────
# "0: Who used the screwdriver first? (Day: DAY1)"
Q_START_RE = re.compile(r"^(?P<idx>\d+):\s+(?P<q>.+?)\s+\(Day:\s+(?P<day>DAY\d+)\)\s*$")

# "After soft ranking: 6 FIOs"
SOFT_RE = re.compile(r"After soft ranking:\s*(?P<n>\d+)\s*FIOs")

# "  [Evidence] top-5 FIOs:"  (start of FIO block)
EVID_HDR_RE = re.compile(r"\[Evidence\]\s+top-(?P<n>\d+)\s+FIOs:")

# "    [0] DAY1 111230 | p=Jake | a=handed | o=screw | w=0.246 | α=0.504 | loc=across from me"
FIO_LINE_RE = re.compile(
    r"^\s*\[(?P<rank>\d+)\]\s+"
    r"(?P<day>DAY\d+)\s+(?P<time>\d+)\s*\|\s*"
    r"(?P<kv>.+?)\s*$"
)

# "  [LLM fallback] evidence=1 items | input_tokens=2044 | context_chars=1304"
LLM_EV_RE = re.compile(r"\[LLM fallback\]\s+evidence=(?P<n>\d+)\s+items")

# "[FALLBACK] No handler matched: ..."
NO_HANDLER_RE = re.compile(r"^\[FALLBACK\]\s+No handler matched")

# "Model Answer: ..."
MODEL_ANS_RE = re.compile(r"^Model Answer:\s*(?P<rest>.*)$")

# "Extracted Option: A"
EXTRACTED_RE = re.compile(r"^Extracted Option:\s*(?P<opt>\S+)\s*$")

# "Correct Answer: B - Alice"
CORRECT_RE = re.compile(
    r"^Correct Answer:\s*(?P<opt>[A-Z])\s*-\s*(?P<text>.*?)\s*$"
)

# "  [Summary Q0] type=EntityLog pred=A correct=B ✗ [grounded]"
SUMMARY_RE = re.compile(
    r"\[Summary Q(?P<idx>\d+)\]\s+type=(?P<qtype>\S+)\s+"
    r"pred=(?P<pred>\S+)\s+correct=(?P<corr>\S+)\s+"
    r"(?P<mark>[✓✗])\s+\[(?P<status>grounded|cascade)\]"
)

# "Explain your choice in one sentence: ..."   OR   "Explain your choice in one sentence. ..."
EXPLAIN_RE = re.compile(
    r"Explain your choice in one sentence[:.]?\s*(?P<rest>.*)$"
)


# ──────────────────────────────────────────────────────────────────────────────
# Parse a single FIO evidence line like
#   "p=Jake | a=handed | o=screw | w=0.246 | α=0.504 | loc=across from me"
# into a dict. Fields that may be missing default to None.
# ──────────────────────────────────────────────────────────────────────────────
def parse_fio_kv(kv_str: str) -> dict:
    out = {
        "person": None,
        "action": None,
        "object": None,
        "w": None,
        "alpha": None,
        "loc": None,
        "speech": None,
        "merged": None,
    }
    # Split on " | " but be careful: speech='Jake said: Haha, you can' may
    # contain a "|" inside quotes? Looking at the log it doesn't, so plain
    # split is fine.
    parts = [p.strip() for p in kv_str.split("|") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        # strip surrounding quotes if present
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        if k == "p":
            out["person"] = v
        elif k == "a":
            out["action"] = v
        elif k == "o":
            out["object"] = None if v == "—" else v
        elif k == "w":
            try:
                out["w"] = float(v)
            except ValueError:
                pass
        elif k in ("α", "alpha"):
            try:
                out["alpha"] = float(v)
            except ValueError:
                pass
        elif k == "loc":
            out["loc"] = v
        elif k == "speech":
            out["speech"] = v
        elif k == "merged":
            try:
                out["merged"] = int(v)
            except ValueError:
                pass
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main streaming parser
# ──────────────────────────────────────────────────────────────────────────────
def parse_log(path: Path) -> list:
    records = []
    cur = None
    state = "idle"          # "idle" | "in_evidence" | "after_model_ans"
    explanation_lines = []  # accumulator for multi-line explanations

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

            # ── new question ───────────────────────────────────────────────
            m = Q_START_RE.match(line)
            if m:
                # close previous
                if cur is not None:
                    flush(cur)
                cur = {
                    "q_idx":          int(m.group("idx")),
                    "qtype":          None,
                    "day":            m.group("day"),
                    "question":       m.group("q").strip(),
                    "gt_option":      None,
                    "gt_text":        None,
                    "pred_option":    None,
                    "correct":        None,
                    "fios_retrieved": 0,
                    "fios_top_k":     [],
                    "llm_evidence_n": None,
                    "path":           None,
                    "explanation":    None,
                    "summary_tag":    None,
                }
                explanation_lines = []
                state = "idle"
                continue

            if cur is None:
                continue

            # ── retrieval count ────────────────────────────────────────────
            m = SOFT_RE.search(line)
            if m:
                cur["fios_retrieved"] = int(m.group("n"))
                continue

            # ── evidence block start ───────────────────────────────────────
            if EVID_HDR_RE.search(line):
                state = "in_evidence"
                continue

            # ── evidence FIO line ──────────────────────────────────────────
            if state == "in_evidence":
                m = FIO_LINE_RE.match(line)
                if m:
                    fio = parse_fio_kv(m.group("kv"))
                    fio["rank"] = int(m.group("rank"))
                    fio["day"]  = m.group("day")
                    fio["time"] = m.group("time")
                    cur["fios_top_k"].append(fio)
                    continue
                # any non-FIO line ends the block
                state = "idle"

            # ── LLM fallback marker ────────────────────────────────────────
            m = LLM_EV_RE.search(line)
            if m:
                cur["llm_evidence_n"] = int(m.group("n"))
                if cur["path"] is None:
                    cur["path"] = "A"
                continue

            # ── no-handler fallback ────────────────────────────────────────
            if NO_HANDLER_RE.search(line):
                cur["path"] = "C"
                continue

            # ── Model Answer ───────────────────────────────────────────────
            m = MODEL_ANS_RE.match(line)
            if m:
                # If the rest of the line already has "Answer: X" or even
                # "Explain your choice..." inline, capture it.
                rest = m.group("rest")
                m2 = EXPLAIN_RE.search(rest)
                if m2 and m2.group("rest"):
                    explanation_lines.append(m2.group("rest"))
                state = "after_model_ans"
                continue

            # ── Explain your choice line (can appear with or without colon) ─
            m = EXPLAIN_RE.search(line)
            if m:
                if m.group("rest"):
                    explanation_lines.append(m.group("rest"))
                state = "after_model_ans"
                continue

            # ── Extracted Option ───────────────────────────────────────────
            m = EXTRACTED_RE.match(line)
            if m:
                cur["pred_option"] = m.group("opt")
                state = "idle"
                continue

            # ── Correct Answer ─────────────────────────────────────────────
            m = CORRECT_RE.match(line)
            if m:
                cur["gt_option"] = m.group("opt")
                cur["gt_text"]   = m.group("text").strip()
                continue

            # ── Summary line: closes the question ──────────────────────────
            m = SUMMARY_RE.search(line)
            if m:
                cur["qtype"]        = m.group("qtype")
                cur["summary_tag"]  = m.group("status")
                cur["correct"]      = (m.group("mark") == "✓")
                # pred from summary if we somehow missed Extracted Option
                if cur["pred_option"] is None:
                    p = m.group("pred")
                    cur["pred_option"] = p if p != "None" else None
                # path inference if not yet set:
                if cur["path"] is None:
                    if cur["summary_tag"] == "cascade":
                        cur["path"] = "cascade"
                    else:
                        cur["path"] = "B"   # deterministic handler
                # cascade tag overrides any earlier path label
                if cur["summary_tag"] == "cascade":
                    cur["path"] = "cascade"
                flush(cur)
                cur = None
                explanation_lines = []
                state = "idle"
                continue

            # ── otherwise: if we're in the "after model ans" zone, treat
            #     non-empty lines as explanation continuation, but only if
            #     they look like prose (not stray log noise).
            if state == "after_model_ans":
                stripped = line.strip()
                if not stripped:
                    continue
                # don't slurp obvious log noise
                if stripped.startswith("[") or stripped.startswith("Using ") \
                        or stripped.startswith("Answer:"):
                    continue
                explanation_lines.append(stripped)

    # last record (in case file doesn't end with a Summary line)
    if cur is not None:
        flush(cur)

    return records


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    recs = parse_log(FIO_LOG)
    OUT.write_text(json.dumps(recs, indent=2, ensure_ascii=False))
    print(f"[ok] wrote {len(recs)} records → {OUT}")

    # quick sanity readout
    from collections import Counter
    paths = Counter(r["path"] for r in recs)
    tags  = Counter(r["summary_tag"] for r in recs)
    types = Counter(r["qtype"] for r in recs)
    n_corr = sum(1 for r in recs if r["correct"])
    n_with_fios = sum(1 for r in recs if r["fios_top_k"])
    n_with_expl = sum(1 for r in recs if r["explanation"])
    print(f"   correct      : {n_corr}/{len(recs)}")
    print(f"   has fios_top_k: {n_with_fios}/{len(recs)}")
    print(f"   has explanation: {n_with_expl}/{len(recs)}")
    print(f"   paths        : {dict(paths)}")
    print(f"   summary_tags : {dict(tags)}")
    print(f"   qtypes       : {dict(types)}")