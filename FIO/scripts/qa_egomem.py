# scripts/qa_egomem.py
# ─────────────────────────────────────────────────────────────────────────────
# Run the EgoMem benchmark (egomem_reason_q.json) on the existing FIO memory
# using the scripts/qa.py pipeline.
#
# Why a wrapper instead of editing scripts/qa.py in place:
#   scripts/qa.py stays untouched (your EgoLifeQA runs remain reproducible). This
#   file loads its source, applies a small set of asserted text patches, and
#   adds the EgoMem-specific parts. Keep it in the same folder as scripts/qa.py.
#
# What differs between EgoLifeQA and EgoMem (and how it's handled here):
#   1. Schema   query_time "DAY7, 19:00:00" (str) -> {"date","time"}
#               options {"A":..} -> choice_a ... choice_j
#               query_type -> pipeline "type" (see TYPE_MAP)
#   2. Options  EgoMem has up to 10 options (A-J); q_a_v6 hardcodes A-D in the
#               prompt, the answer regex and every handler -> patched to LETTERS
#   3. Identity 6 wearers (A1_JAKE ... A6_SHURE), one FIO memory each; the
#               DB is staged/opened and the camera wearer set per identity
#   4. Types    Event Ordering (sequence options) gets its own handler:
#               per-event retrieval -> timestamps -> best-matching option
#               Counting / state-tracking questions go straight to the LLM
#               (the deterministic option-matchers don't make sense for them)
#   5. Labels   the json has NO "answer" field -> predictions are written to a
#               JSONL; accuracy is computed only if "answer" is present
#
# Usage (from FIO/):
#   python scripts/qa_egomem.py --plan                 # memory + progress per person
#   python scripts/qa_egomem.py --wearer jake          # one person (or jake,tasha / all)
#   python scripts/qa_egomem.py --export               # build submission.json for the Space
#   Each person reads its own memory, FIO_mem/<IDENTITY>_full by default
#   (--memory-template / --memory IDENTITY=PATH to change). Predictions go to
#   results/egomem/<run>/<IDENTITY>.jsonl; when all six are done, submission.json
#   ([{"example_id", "predicted_answer"}] x 500) is written next to them for
#   https://huggingface.co/spaces/Ted412/EgoMemReason.
#
# Env knobs:
#   EGOMEM_QUESTIONS   questions json  (default benchmarks/QA/egomem_reason_qa.json)
#   EGOMEM_DB_TEMPLATE memory folder per person, placeholders {identity} {wearer} {build}
#   EGOMEM_BUILD       value for {build} in the template
#   EGOMEM_WEARER      all | jake | A1_JAKE | comma list (default all); or CLI --wearer
#   EGOMEM_RUN         run name (default from template, DB mode, K, temperature); or --run-name
#   EGOMEM_RESULTS_DIR parent of the run folders (default FIO/results/egomem)
#   EGOMEM_QTYPES      comma list of query_type values         (default all; partial run)
#   EGOMEM_LLM_ONLY    query_types routed straight to the LLM
#                      (default "Cumulative State Tracking,Temporal Counting")
#   EGOMEM_ORDER_MODE  hybrid | heuristic | llm   (Event Ordering, default hybrid)
#   EGOMEM_LIMIT       max questions per person (debugging; partial run)
#   BUILD_CONFIG, DB_MODE, EVIDENCE_K, QA_TEMPERATURE, QA_RANKER, SEED, LOCAL_DB_ROOT  as in scripts/qa.py
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import re
import sys
import types
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))         # FIO/scripts/
FIO_ROOT = os.path.dirname(HERE)                          # FIO/  (holds the `fio` package)
REPO_ROOT = os.path.dirname(FIO_ROOT)                     # EgoLife_Research/ — same on cluster and sshfs mount
V6_PATH = os.path.join(HERE, "qa.py")                     # the canonical QA pipeline this wraps
if FIO_ROOT not in sys.path:
    sys.path.insert(0, FIO_ROOT)                          # so `import fio` works from any directory

LETTERS = list("ABCDEFGHIJ")
LETTERS_L = [c.lower() for c in LETTERS]

TYPE_MAP = {
    "Activity Pattern":          "HabitInsight",
    "Spatial Preference":        "HabitInsight",
    "Event Linking":             "EventRecall",
    "Cumulative State Tracking": "EventRecall",
    "Temporal Counting":         "EventRecall",
    "Event Ordering":            "EventOrdering",
}
LLM_ONLY_QTYPES = {s.strip() for s in os.environ.get(
    "EGOMEM_LLM_ONLY", "Cumulative State Tracking,Temporal Counting").split(",") if s.strip()}
LLM_ONLY_TYPE_NAME = "EgoMemLLMOnly"      # pipeline "type" given to those questions
ORDER_MODE = os.environ.get("EGOMEM_ORDER_MODE", "hybrid")
ORDER_MIN_SIM = float(os.environ.get("EGOMEM_ORDER_MIN_SIM", "0.40"))
ORDER_MIN_MATCHED = float(os.environ.get("EGOMEM_ORDER_MIN_MATCHED", "0.8"))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load q_a_v6 with the A-D -> A-J patches
# ══════════════════════════════════════════════════════════════════════════════

def _sub(src: str, old: str, new: str, min_count: int = 1, label: str = "") -> str:
    n = src.count(old)
    if n < min_count:
        raise RuntimeError(f"[patch] '{label or old[:50]}' found {n}x, expected >= {min_count}. "
                           f"qa.py differs from the version this wrapper targets.")
    print(f"[patch] {label or old[:40]!r}: {n} replacement(s)")
    return src.replace(old, new)


def load_v6() -> types.ModuleType:
    with open(V6_PATH, "r") as f:
        src = f.read()

    # option-letter literals used by handlers
    src = _sub(src, '["A", "B", "C", "D"]', "LETTERS", 4, "option letters (upper)")
    src = _sub(src, '["a","b","c","d"]', "LETTERS_L", 1, "option letters (lower list)")
    src = _sub(src, 'for c in "abcd"', "for c in LETTERS_L", 1, "option letters (abcd loop)")
    src = _sub(src, '{"A", "B", "C", "D"}', "set(LETTERS)", 1, "format_mcq_answer set")
    # answer-letter regexes (extract_single_option_answer)
    src = _sub(src, "([A-D])", "([A-J])", 2, "answer regex")

    # LLM prompt: list every option, don't restrict to A-D
    src = _sub(src, '''A. {question.get("choice_a")}
B. {question.get("choice_b")}
C. {question.get("choice_c")}
D. {question.get("choice_d")}''', "{_choices_block(question)}", 1, "prompt choices")
    src = _sub(src, "choose from A, B, C, D only", "choose only from the option letters listed above", 1, "prompt letters")
    src = _sub(src, "Answer: <A/B/C/D>", "Answer: <option letter>", 1, "prompt format")

    # LLM-only question types skip the deterministic handlers
    src = _sub(src, '    if question_type == "EntityLog":',
               '    if question_type == LLM_ONLY_TYPE_NAME:\n'
               '        return llm_fallback_answer(question, q_json, _llm_evidence(ranked), question_info)\n\n'
               '    if question_type == "EntityLog":', 1, "LLM-only guard")

    mod = types.ModuleType("q_a_v6_egomem")
    mod.__file__ = V6_PATH
    mod.LETTERS, mod.LETTERS_L = LETTERS, LETTERS_L
    mod.LLM_ONLY_TYPE_NAME = LLM_ONLY_TYPE_NAME

    def _choices_block(question: Dict) -> str:
        return "\n".join(f"{c}. {question.get(f'choice_{c.lower()}')}"
                         for c in LETTERS if question.get(f"choice_{c.lower()}"))
    mod._choices_block = _choices_block

    sys.modules["q_a_v6_egomem"] = mod
    sys.path.insert(0, FIO_ROOT)
    exec(compile(src, V6_PATH, "exec"), mod.__dict__)
    return mod


# ══════════════════════════════════════════════════════════════════════════════
# 2. Schema normalisation
# ══════════════════════════════════════════════════════════════════════════════

_COUNT_RE = re.compile(r"\bhow many\b|\bhow has the number\b|\bhow long\b", re.I)

def normalize_question(q: Dict) -> Dict:
    out = dict(q)
    qt = q.get("query_time", "")
    if isinstance(qt, str):
        m = re.match(r"\s*(DAY\d+)\s*,\s*(\d{1,2}):(\d{2}):(\d{2})", qt)
        if not m:
            raise ValueError(f"unparseable query_time {qt!r}")
        out["query_time"] = {"date": m.group(1),
                             "time": f"{int(m.group(2)):02d}{m.group(3)}{m.group(4)}"}
    for letter, text in (q.get("options") or {}).items():
        out[f"choice_{letter.lower()}"] = text
    out["_orig_type"] = q.get("query_type", "")
    if q.get("query_type") in LLM_ONLY_QTYPES:
        out["type"] = LLM_ONLY_TYPE_NAME
    else:
        out["type"] = TYPE_MAP.get(q.get("query_type"), "EventRecall")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. Event Ordering handler
# ══════════════════════════════════════════════════════════════════════════════

_EVENT_RE = re.compile(r"^\s*(\d+)\)\s*(.+?)\s*$", re.M)

def _abs_minutes(mod, meta: Dict) -> float:
    return (mod.normalize_date(meta.get("date", "DAY1")) * 1440.0
            + (mod.parse_hhmmss_to_mins(meta.get("start_time", 0)) or 0.0))

def _match_events(mod, events: Dict[int, str], storage, db: str, qinfo: Dict) -> Dict[int, Optional[Dict]]:
    """For each event, retrieve FIOs and pick the best-timed match."""
    matches: Dict[int, Optional[Dict]] = {}
    for idx, text in events.items():
        q_json = {"entity": "", "action": "", "query_intent": "when", "_question_text": text}
        try:
            res = storage.search(query=q_json, k=30, query_day=qinfo["day"], db=db)
            metas = (res.get("metadatas") or [[]])[0]
        except Exception as e:
            print(f"  [order] search failed for event {idx}: {e}")
            metas = []
        metas = mod.filter_by_query_time(metas, qinfo)
        if not metas:
            matches[idx] = None
            continue
        docs = [" ".join(str(m.get(f, "")) for f in
                         ("action", "object", "tool", "app", "speech_content",
                          "source_location", "target_location")).strip() or "x"
                for m in metas]
        embs = np.array(mod._embed_model([text] + docs))
        sims = [mod.cosine(embs[0], embs[i + 1]) for i in range(len(metas))]
        best = max(sims)
        strong = [i for i, s in enumerate(sims) if s >= best - 0.05]
        tl = text.lower()
        if "first time" in tl:
            pick = min(strong, key=lambda i: _abs_minutes(mod, metas[i]))
        elif "last time" in tl:
            pick = max(strong, key=lambda i: _abs_minutes(mod, metas[i]))
        else:
            pick = max(strong, key=lambda i: sims[i])
        matches[idx] = {"meta": metas[pick], "sim": sims[pick], "t": _abs_minutes(mod, metas[pick])}
    return matches

def _score_option(opt_seq: List[int], matches: Dict[int, Optional[Dict]], subset_variant: bool) -> float:
    timed = [e for e in opt_seq if matches.get(e)]
    pairs = conc = 0
    for i in range(len(timed)):
        for j in range(i + 1, len(timed)):
            pairs += 1
            conc += matches[timed[i]]["t"] <= matches[timed[j]]["t"]
    score = conc / pairs if pairs else 0.0
    if subset_variant:   # reward including exactly the events that have happened
        happened = {e for e, m in matches.items() if m and m["sim"] >= ORDER_MIN_SIM}
        score -= 0.15 * len(set(opt_seq) - happened)
        score -= 0.15 * len(happened - set(opt_seq))
    return score

def answer_event_ordering(mod, q: Dict, storage, db: str) -> str:
    qinfo = {"day": q["query_time"]["date"], "time": q["query_time"]["time"]}
    events = {int(i): t for i, t in _EVENT_RE.findall(q["question"])}
    subset_variant = "haven't happened yet" in q["question"].lower()

    matches = _match_events(mod, events, storage, db, qinfo) if events else {}
    n_ok = sum(1 for m in matches.values() if m and m["sim"] >= ORDER_MIN_SIM)
    frac = n_ok / len(events) if events else 0.0
    print(f"  [order] events={len(events)} matched={n_ok} subset_variant={subset_variant}")

    if ORDER_MODE != "llm" and events and (ORDER_MODE == "heuristic" or
                                           (frac >= ORDER_MIN_MATCHED or subset_variant and n_ok >= 2)):
        scored = []
        for letter in LETTERS:
            text = q.get(f"choice_{letter.lower()}")
            if text:
                scored.append((_score_option([int(x) for x in re.findall(r"\d+", text)],
                                             matches, subset_variant), letter))
        if scored:
            scored.sort(reverse=True)
            print(f"  [order] top options: {scored[:3]}")
            return mod.format_mcq_answer(
                scored[0][1], f"Event timestamps ordered; best concordance {scored[0][0]:.2f}.")

    # LLM fallback with per-event timestamped evidence (K must cover every event)
    evidence = []
    for idx, text in events.items():
        m = matches.get(idx)
        ev = {"event_no": idx, "event": text}
        if m:
            fm = m["meta"]
            ev.update({"date": fm.get("date"), "start_time": str(fm.get("start_time", ""))[:6],
                       "matched_action": fm.get("action"), "matched_object": fm.get("object"),
                       "speech": (fm.get("speech_content") or "")[:80],
                       "similarity": round(m["sim"], 3)})
        else:
            ev["note"] = "no matching record before query time"
        evidence.append(ev)
    old_k = mod.EVIDENCE_K
    mod.EVIDENCE_K = max(old_k, len(evidence))
    try:
        return mod.llm_fallback_answer(
            q, {"query_intent": "order the events from earliest to latest"}, evidence, qinfo)
    finally:
        mod.EVIDENCE_K = old_k


# ══════════════════════════════════════════════════════════════════════════════
# 4. Driver
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 4. Per-person runs, results layout and the leaderboard submission
# ══════════════════════════════════════════════════════════════════════════════
#
# results/egomem/<run>/<IDENTITY>.jsonl         one prediction per question (resumable)
# results/egomem/<run>/<IDENTITY>.config.json   settings + environment of that person's run
# results/egomem/<run>/submission.json          upload this to the EgoMemReason Space
# results/egomem/<run>/submission_report.json   what was filled in, per person / type

DEFAULT_TEMPLATE = os.path.join(REPO_ROOT, "FIO_mem", "{identity}_full")


def _resolve_wearers(arg: str, all_ids: List[str]) -> List[str]:
    """all | jake | A1_JAKE | comma list (names and ids, case-insensitive)."""
    tokens = [t.strip().lower() for t in arg.split(",") if t.strip()]
    if not tokens or "all" in tokens:
        return list(all_ids)
    picked = [i for i in all_ids if i.lower() in tokens or i.split("_", 1)[1].lower() in tokens]
    unknown = [t for t in tokens if not any(t in (i.lower(), i.split("_", 1)[1].lower()) for i in all_ids)]
    if unknown or not picked:
        raise SystemExit(f"Unknown wearer(s) {unknown or tokens}. Available: all, "
                         + ", ".join(all_ids) + " (or their names: "
                         + ", ".join(i.split('_', 1)[1].lower() for i in all_ids) + ")")
    return picked


def _memory_roots(all_ids: List[str], template: str, build: str, overrides: List[str]) -> Dict[str, str]:
    roots = {i: template.format(build=build, identity=i, wearer=i.split("_", 1)[1].lower()) for i in all_ids}
    for o in overrides:
        if "=" not in o:
            raise SystemExit(f"--memory expects IDENTITY=PATH, got {o!r}")
        who, path = o.split("=", 1)
        ident = _resolve_wearers(who, all_ids)
        if len(ident) != 1:
            raise SystemExit(f"--memory {o!r}: name exactly one person")
        roots[ident[0]] = os.path.abspath(os.path.expanduser(path))
    return roots


def _run_name(template: str, db_mode: str, k: int, temp: float) -> str:
    tag = os.path.basename(os.path.normpath(template)).replace("{identity}", "").replace("{wearer}", "")
    tag = tag.replace("{build}", "build").strip("_-") or "memory"
    return f"{tag}_{db_mode}_K{k}_T{temp:g}"


def _read_preds(path: str) -> Dict[str, Dict]:
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["p_id"]] = r
    return out


def export_submission(run_dir: str, questions: List[Dict], allow_missing: bool = False) -> Optional[str]:
    """Merge every person's predictions into submission.json in the Space's format:
    [{"example_id": int, "predicted_answer": letter}] — exactly one per question,
    each letter one of that question's options. Questions without a valid
    prediction are filled with the question's first option and counted."""
    from fio.repro import write_json
    by_id = defaultdict(list)
    for q in questions:
        by_id[q["identity"]].append(q)
    preds = {}
    missing_people = []
    for ident in sorted(by_id):
        got = _read_preds(os.path.join(run_dir, f"{ident}.jsonl"))
        if not got:
            missing_people.append(ident)
        preds.update(got)
    if missing_people and not allow_missing:
        print(f"[export] not written: no predictions yet for {missing_people}. Run them "
              f"(--wearer {','.join(i.split('_', 1)[1].lower() for i in missing_people)}), "
              f"or pass --allow-missing to fill them with placeholders.")
        return None

    submission, filled = [], defaultdict(lambda: defaultdict(int))
    for q in sorted(questions, key=lambda x: x["example_id"]):
        letters = sorted(k.strip().upper() for k in q["options"])
        pred = (preds.get(q["p_id"], {}).get("pred") or "").strip().upper()
        if pred not in letters:
            reason = "not run" if q["p_id"] not in preds else "no answer"
            filled[q["identity"]][f"{reason}: {q['query_type']}"] += 1
            pred = letters[0]
        submission.append({"example_id": int(q["example_id"]), "predicted_answer": pred})

    # Same checks as the Space's evaluator.py: int ids, no duplicates, full coverage, valid letters.
    ids = [s["example_id"] for s in submission]
    assert all(type(i) is int for i in ids), "example_id must be int"
    assert len(ids) == len(set(ids)) == len(questions), "duplicate or missing example_id"
    opts = {int(q["example_id"]): {k.strip().upper() for k in q["options"]} for q in questions}
    assert all(s["predicted_answer"] in opts[s["example_id"]] for s in submission), "invalid letter"

    sub_path = os.path.join(run_dir, "submission.json")
    with open(sub_path, "w") as f:
        json.dump(submission, f, indent=1)
    n_filled = sum(sum(v.values()) for v in filled.values())
    write_json(os.path.join(run_dir, "submission_report.json"), {
        "n_questions": len(submission), "n_predicted": len(submission) - n_filled, "n_filled": n_filled,
        "fill_rule": "first option letter of the question", "filled": {k: dict(v) for k, v in filled.items()},
        "configs": {i: json.load(open(c)).get("config") for i in sorted(by_id)
                    if os.path.exists(c := os.path.join(run_dir, f"{i}.config.json"))},
    })
    print(f"[export] {sub_path}: {len(submission)} answers, {n_filled} filled with a placeholder"
          + (f" ({ {k: sum(v.values()) for k, v in filled.items()} })" if n_filled else ""))
    print("         upload it at https://huggingface.co/spaces/Ted412/EgoMemReason")
    return sub_path


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="EgoMemReason QA on per-person FIO memories + leaderboard submission.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
examples:
  python scripts/qa_egomem.py --plan                       # memory + progress per person
  python scripts/qa_egomem.py --wearer jake                # one person
  python scripts/qa_egomem.py --wearer jake,tasha          # several
  python scripts/qa_egomem.py --wearer all                 # everyone, then writes submission.json
  python scripts/qa_egomem.py --memory A6_SHURE=../FIO_mem/A6_SHURE_RE_full --wearer shure
  python scripts/qa_egomem.py --export                     # (re)build submission.json from finished runs
""")
    ap.add_argument("--wearer", default=None,
                    help="all | jake | A1_JAKE | comma list (default: env EGOMEM_WEARER or all)")
    ap.add_argument("--memory-template", default=None,
                    help="memory folder per person; placeholders {identity} {wearer} {build} "
                         "(default: env EGOMEM_DB_TEMPLATE or FIO_mem/{identity}_full)")
    ap.add_argument("--memory", action="append", default=[], metavar="IDENTITY=PATH",
                    help="override one person's memory folder (repeatable)")
    ap.add_argument("--run-name", default=None,
                    help="results/egomem/<run-name>/ (default: from the template, DB mode, K and temperature)")
    ap.add_argument("--plan", action="store_true", help="show memory and progress per person, run nothing")
    ap.add_argument("--export", action="store_true", help="only (re)build submission.json")
    ap.add_argument("--allow-missing", action="store_true",
                    help="write submission.json even if some people have no predictions (placeholders)")
    args = ap.parse_args()

    qpath = os.environ.get("EGOMEM_QUESTIONS",
                           os.path.join(REPO_ROOT, "benchmarks", "QA", "egomem_reason_qa.json"))
    with open(qpath) as f:
        questions = json.load(f)
    all_ids = sorted({q["identity"] for q in questions})
    db_mode = os.environ.get("DB_MODE", "opt")
    evidence_k = int(os.environ.get("EVIDENCE_K", "5"))
    qa_temp = float(os.environ.get("QA_TEMPERATURE", "0"))
    template = args.memory_template or os.environ.get("EGOMEM_DB_TEMPLATE", DEFAULT_TEMPLATE)
    build = os.environ.get("EGOMEM_BUILD", "")
    roots = _memory_roots(all_ids, template, build, args.memory)
    run_name = args.run_name or os.environ.get("EGOMEM_RUN") or _run_name(template, db_mode, evidence_k, qa_temp)
    run_dir = os.path.join(os.environ.get("EGOMEM_RESULTS_DIR", os.path.join(FIO_ROOT, "results", "egomem")), run_name)
    need_db = "chroma_fio_opt" if db_mode == "opt" else "chroma_fio_raw"
    wearers = _resolve_wearers(args.wearer or os.environ.get("EGOMEM_WEARER")
                               or os.environ.get("EGOMEM_IDENTITIES") or "all", all_ids)
    n_per = defaultdict(int)
    for q in questions:
        n_per[q["identity"]] += 1

    if args.plan or args.export:
        print(f"run: {run_dir}")
        for i in all_ids:
            has = os.path.isdir(os.path.join(roots[i], need_db))
            done = len(_read_preds(os.path.join(run_dir, f"{i}.jsonl")))
            print(f"  {i:11} questions={n_per[i]:3d} done={done:3d}  memory={'ok     ' if has else 'MISSING'} {roots[i]}")
        if args.export:
            export_submission(run_dir, questions, args.allow_missing)
        return

    from fio.repro import run_info, set_seed, write_json
    seed = set_seed()
    print(f"[repro] seed = {seed}")
    _IMP_PRESENT = {"full", "no_s3", "no_ctx", "no_freq", "no_recency", "no_distinct"}
    _IMP_ABSENT = {"no_s2", "no_s1", "s3_only"}

    def ranker_ablation_for(bc: str) -> str:
        bc = bc.strip().lower()
        if bc in _IMP_PRESENT: return "full"
        if bc in _IMP_ABSENT:  return "no_imp"
        raise ValueError(f"Unknown build config {bc!r}")

    mod = load_v6()
    mod.ABLATION = (os.environ.get("QA_RANKER")                                  # qa.py reads this global
                    or ranker_ablation_for(os.environ.get("BUILD_CONFIG", "full")))
    want_qt = {s.strip() for s in os.environ.get("EGOMEM_QTYPES", "").split(",") if s.strip()}
    limit = int(os.environ["EGOMEM_LIMIT"]) if os.environ.get("EGOMEM_LIMIT") else None

    from fio.storage.staging import StagedChromaDB
    stager = StagedChromaDB(local_root=os.environ.get("LOCAL_DB_ROOT"), sync_on_exit=False)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[run] {run_dir}\n[wearer] {wearers}")

    results = []   # (query_type, pred, answer_key, grounded)
    for identity in wearers:
        wearer = identity.split("_", 1)[1]
        root = roots[identity]
        opt_dir = os.path.join(root, "chroma_fio_opt")
        raw_dir = os.path.join(root, "chroma_fio_raw")
        if not os.path.isdir(os.path.join(root, need_db)):
            print(f"\n[SKIP] {identity}: no memory at {os.path.join(root, need_db)}. "
                  f"Build it, or point --memory {identity}=<folder> at the right one.")
            continue

        out_path = os.path.join(run_dir, f"{identity}.jsonl")
        cfg_path = os.path.join(run_dir, f"{identity}.config.json")
        run_cfg = {
            "identity": identity, "memory": root, "questions": qpath,
            "db_mode": db_mode, "build_config": os.environ.get("BUILD_CONFIG", "full"), "ranker": mod.ABLATION,
            "evidence_k": mod.EVIDENCE_K, "qa_temperature": mod.QA_TEMPERATURE, "seed": seed,
            "qtypes": sorted(want_qt), "limit": limit, "llm_only_qtypes": sorted(LLM_ONLY_QTYPES),
            "order_mode": ORDER_MODE,
            "llm_model": mod.Q_decompose.MODEL_ID, "llm_revision": mod.Q_decompose.MODEL_REVISION,
        }
        if os.path.exists(out_path) and os.path.exists(cfg_path):
            with open(cfg_path) as f:
                prev = json.load(f).get("config", {})
            diff = {k: (prev.get(k), v) for k, v in run_cfg.items() if prev.get(k) != v}
            if diff:
                raise SystemExit(f"[resume] {out_path} was made with different settings {diff}. "
                                 "Use another --run-name, or delete that person's file.")
        write_json(cfg_path, {"config": run_cfg, "env": run_info()})

        qs = [q for q in questions if q["identity"] == identity and (not want_qt or q["query_type"] in want_qt)]
        if limit is not None:
            qs = qs[:limit]
        done = _read_preds(out_path)
        todo = [q for q in qs if q["p_id"] not in done]
        if done:
            print(f"[resume] {identity}: {len(done)} done, {len(todo)} to go")
        if not todo:
            continue

        mod.set_camera_wearer(wearer)
        # distinct staging name per identity so staged copies don't collide
        opt_local = stager.stage_in(f"chroma_fio_opt_{identity}", opt_dir if db_mode == "opt" else None)
        raw_local = stager.stage_in(f"chroma_fio_raw_{identity}", raw_dir if db_mode == "raw" else None)
        storage = mod.DualFIOStorage(raw_db_dir=raw_local or raw_dir, opt_db_dir=opt_local or opt_dir,
                                     store_raw=(db_mode == "raw"), store_opt=(db_mode == "opt"))

        print(f"\n{'='*80}\n{identity}: {len(todo)} questions | memory={root}\n{'='*80}")
        with open(out_path, "a") as fout:
            for i, rq in enumerate(todo):
                q = normalize_question(rq)
                print(f"\n[{identity} {i}] {rq['p_id']} ({rq['query_type']}) {q['question'][:90]!r}")
                mod._ANSWER_TRACE.update(llm=False, no_handler=False)
                try:
                    if rq["query_type"] == "Event Ordering":
                        answer = answer_event_ordering(mod, q, storage, db_mode)
                    else:
                        answer = mod.q_a_v2(question=q, dual_storage=storage, db=db_mode)
                except Exception as e:
                    print(f"[ERROR] {rq['p_id']}: {e}")
                    answer = None
                pred = mod.extract_single_option_answer(answer) if answer else None
                if pred and not q.get(f"choice_{pred.lower()}"):
                    pred = None                      # letter not among this question's options
                non_refusal = bool(pred) and "cannot be determined" not in (answer or "").lower()
                path = mod.answer_path(pred, mod._ANSWER_TRACE["llm"], mod._ANSWER_TRACE["no_handler"])
                grounded = mod.classify_grounding(path, pred, answer or "")["grounded"]
                key = rq.get("answer")
                rec = {"example_id": rq["example_id"], "p_id": rq["p_id"], "identity": identity,
                       "query_type": rq["query_type"], "pred": pred, "path": path, "grounded": grounded,
                       "non_refusal": non_refusal, "answer_text": answer, "gold": key,
                       "evidence_k": mod.EVIDENCE_K}
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                results.append((rq["query_type"], pred, key, grounded))
                print(f"  -> pred={pred} path={path} {'[grounded]' if grounded else '[not grounded]'}"
                      + (f" gold={key} {'✓' if pred == key else '✗'}" if key else ""))

    # summary of this invocation
    print(f"\n{'='*80}\nSUMMARY (K={mod.EVIDENCE_K}) -> {run_dir}\n{'='*80}")
    per = defaultdict(lambda: [0, 0, 0, 0])   # total, answered, correct, grounded correct
    for qt, pred, key, grounded in results:
        per[qt][0] += 1
        per[qt][1] += pred is not None
        per[qt][2] += (key is not None and pred == key)
        per[qt][3] += (key is not None and pred == key and grounded)
    has_gold = any(r[2] is not None for r in results)
    for qt, (t, a, c, gc) in sorted(per.items()):
        print(f"  {qt:28s} n={t:3d} answered={a:3d}"
              + (f" acc={c/t*100:.1f}% GA={gc/c*100 if c else 0:.1f}%" if has_gold else ""))
    if not has_gold:
        print("  (no 'answer' field in the questions file -> score submission.json on the Space)")

    # progress over all people in this run, and the submission once everyone is done
    print()
    complete = True
    for i in all_ids:
        n_done = len(_read_preds(os.path.join(run_dir, f"{i}.jsonl")))
        complete &= n_done >= n_per[i]
        print(f"  {i:11} {n_done:3d}/{n_per[i]} predicted")
    if want_qt or limit is not None:
        print("[export] skipped: EGOMEM_QTYPES / EGOMEM_LIMIT run is partial by design")
    elif complete:
        export_submission(run_dir, questions)
    else:
        print("[export] submission.json is written once every person is done "
              "(or run --export --allow-missing for a partial one)")


if __name__ == "__main__":
    main()