"""Grounded Accuracy (GA) as defined in the paper (Imprint, Sec. 4.1).

    GA = # grounded correct answers / # correct answers

An answer is *grounded* when it comes from retrieved interaction records:

  path "B"        deterministic handler read the answer off retrieved records
                  → grounded by construction (if an option was committed)
  path "A" / "C"  the LLM wrote the answer from the retrieved evidence
                  (C = no handler matched the question type/intent)
                  → grounded iff its explanation contains none of the
                    self-admission patterns below (no-evidence admissions,
                    hedging, choice elimination)
  path "cascade"  no option committed → never grounded

The patterns and rules are copied verbatim from analysis/ga_old.py, the
log-based analyser used for the paper's numbers; scripts/qa.py now applies
them while answering, so no log parsing is needed.
"""

import re
from collections import defaultdict
from typing import Dict, Iterable, Optional

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


def answer_path(pred: Optional[str], used_llm: bool, no_handler: bool) -> str:
    """Same precedence as the analyser: cascade > C > A > B."""
    if pred is None:
        return "cascade"
    if no_handler:
        return "C"
    if used_llm:
        return "A"
    return "B"


def classify(path: str, pred: Optional[str], explanation: str) -> Dict[str, bool]:
    committed = pred is not None and pred != "None"
    out = {"grounded": False, "self_admitted": False}
    if path == "cascade" or not committed:
        return out
    if path == "B":
        out["grounded"] = True
        return out
    out["self_admitted"] = bool(SELF_ADMIT_RE.search(explanation or ""))
    out["grounded"] = not out["self_admitted"]
    return out


def summarize(records: Iterable[Dict], by: str = "type") -> Dict[str, Dict]:
    """Per-group and overall metrics. Each record needs: correct, grounded,
    path and the grouping key. Keys follow the paper: accuracy = correct / N,
    GA = grounded correct / correct; ER = grounded / N (evidence rate)."""
    groups = defaultdict(list)
    for r in records:
        groups[r.get(by, "?")].append(r)
        groups["overall"].append(r)
    out = {}
    for g, rs in groups.items():
        n = len(rs)
        n_correct = sum(r["correct"] for r in rs)
        n_grounded = sum(r["grounded"] for r in rs)
        n_gc = sum(r["correct"] and r["grounded"] for r in rs)
        paths = defaultdict(int)
        for r in rs:
            paths[r["path"]] += 1
        out[g] = {
            "N": n, "correct": n_correct, "grounded": n_grounded, "grounded_correct": n_gc,
            "accuracy": n_correct / n if n else 0.0,
            "GA": n_gc / n_correct if n_correct else 0.0,
            "ER": n_grounded / n if n else 0.0,
            "paths": dict(paths),
        }
    return out
