# FIO — QA Pipeline

Question answering over first-person video memory built as **FIO** (fine-grained interaction object) records in ChromaDB. This folder holds one canonical QA pipeline and a thin EgoMem driver on top of it.

```
FIO/
├── scripts/qa.py          # canonical pipeline (from q_a_v62706.py)
└── scripts/qa_egomem.py   # EgoMem benchmark wrapper, loads scripts/qa.py at runtime
```

| File | Benchmark | Ranker | Called by |
| --- | --- | --- | --- |
| `scripts/qa.py` | EgoLifeQA (A–D options) | 3-term: entity, action, importance | run directly (see Run) |
| `scripts/qa_egomem.py` | EgoMem (A–J options, 6 wearers) | same as `scripts/qa.py` | `jobs/qw.sh` |

---

## 1. `scripts/qa.py` — canonical pipeline

### Pipeline per question

1. **Parse** `query_time` → `{day, time}`.
2. **Contextual pre-processor** — splits "Now X does Y, what …?" into context anchor + question.
3. **Decompose** with Qwen (`Q_decompose.call_qwen`) into JSON slots: `entity`, `action`, `query_intent`, `temporal_relation`, `related_person`, …
4. **Retrieve** top-30 FIOs from `DualFIOStorage` (date-scoped collection):
   - `RelationMap` → `search_person_filter`
   - `HabitInsight` speech questions → `search_speech_filter`
   - everything else → `search`
5. **Temporal filter** — drop FIOs after the query time.
6. **Soft rank** (skipped for `first_time` / `last_time`):

   ```
   score = w_e · entity_match + w_a · action_match + w_imp · importance_score
   ```

   | `ABLATION` | w_e | w_a | w_imp |
   | --- | --- | --- | --- |
   | `full` | 0.53 | 0.29 | 0.18 |
   | `no_e` | 0.00 | 0.62 | 0.38 |
   | `no_a` | 0.75 | 0.00 | 0.25 |
   | `no_imp` | 0.65 | 0.35 | 0.00 |
   | `equal` | 0.34 | 0.33 | 0.33 |
   | `only_e` | 1.00 | 0.00 | 0.00 |


7. **Temporal relation** — `temporal_candidate_pool()` narrows to FIOs matching the queried entity / action / person / "today" / "before ⟨event⟩" anchor, then `apply_temporal_relation()` takes the first/last occurrence and keeps `EVIDENCE_K` neighbours.
8. **Type-specific handlers** (deterministic first, LLM fallback second):

   | Type | Deterministic handler |
   | --- | --- |
   | `EntityLog` | `when` → calendar-aware Δt (`handle_when`); `who` → qualifier-aware `who_answer`; `where` → source/target location; `what` → object/tool/app |
   | `RelationMap` | `handle_cooccurrence_v2` (giver / anchor person / action conditioned), then `handle_cooccurrence` |
   | `HabitInsight` | `handle_frequency` (speech mode or frequency count) |
   | `TaskMaster` | object/action match, speech evidence to LLM |
   | `EventRecall` | `when` / `what` / `who` resolved deterministically; others to LLM |

9. **LLM fallback** — `llm_fallback_answer()` builds the MCQ prompt from at most `EVIDENCE_K` evidence items. In `opt` mode, merged FIOs are expanded into their `action_sequence` steps first (`expand_opt_evidence`).


### Run

```bash
QA_BUILD_DIR=../FIO_mem/A1_JAKE_full DB_MODE=opt BUILD_CONFIG=full MAX_DAY=7 EVIDENCE_K=5 python -u scripts/qa.py
```

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `QA_BUILD_DIR` | `FIO_mem/A1_JAKE_full` | Build to evaluate: the folder holding `chroma_fio_opt/` (and `chroma_fio_raw/`) |
| `QA_DATA_PATH` | `benchmarks/QA/EgoLifeQA_A1_JAKE.json` | Question file |
| `QA_WEARER` | `Jake` | Camera wearer the questions are filtered to |
| `QA_RANKER` | from `BUILD_CONFIG` | Eq. 3 weights: `full` (0.53, 0.29, 0.18), `no_e`, `no_a`, `no_imp`, `equal`, `only_e` (paper Table 8) |
| `QA_TEMPERATURE` | `0` | LLM decoding: `0` = greedy (deterministic); `0.6` reproduces older sampled runs |
| `SEED` | `1234` | Seeds Python, NumPy and torch |
| `QA_RESULTS` | `FIO/results/qa_<build>_<exp>_<time>.json` | Results file (see Output) |
| `EVIDENCE_K` | `5` | Max evidence items per LLM call (evidence-count ablation) |
| `DB_MODE` | `opt` | `opt` = merged FIO DB, `raw` = unmerged baseline |
| `BUILD_CONFIG` | `full` | Memory build variant → ranker: `full`, `no_s3` → `full`; `no_s1`, `no_s2`, `s3_only` → `no_imp` |
| `MEM_WINDOW` | `day1-7` | Label used in the experiment name |
| `MAX_DAY` | `7` | Evaluate questions from DAY1 … DAY`MAX_DAY` |
| `LOCAL_DB_ROOT` | unset | Node-local path to stage the Chroma DB into (`fio.storage.staging.StagedChromaDB`) |

Evidence-K sweep:

```bash
for K in 2 5 10 15 20; do EVIDENCE_K=$K DB_MODE=opt python -u scripts/qa.py; done
```

Paths are relative to `EgoLife_Research/`.

### Output

`QA_RESULTS` (JSON) holds:
- `config`: build, questions, wearer, DB mode, ranker, K, temperature, seed, LLM model and revision
- `metrics`: `accuracy` and `GA` as in the paper, plus answer rate, accuracy on answered, and per-category `accuracy` / `GA` / `ER` (paper Table 4)
- `questions`: one record per question: prediction, gold, correct, `path`, `grounded`, `self_admitted`, `non_refusal`, raw answer

**GA** (paper Sec. 4.1) = grounded correct / correct, from `fio/grounding.py`. Each answer's `path` is recorded while answering: `B` = a deterministic handler read it off retrieved records (grounded); `A` = the LLM answered from the evidence, `C` = no handler matched and the LLM answered (both grounded only if the explanation has no hedging / no-evidence / choice-elimination language); `cascade` = no option (never grounded). The rules are the ones `analysis/ga_old.py` applied to logs, which no longer needs to be run.
- `env`: git commit (and whether the tree was dirty), package versions, model snapshots, host, Slurm job

The same metrics are printed at the end, and the experiment log goes to `FIO/experiments/{BUILD_CONFIG}_{MEM_WINDOW}_K{EVIDENCE_K}/<time>/` (`EXP_ROOT` overrides the root).

| Metric | Definition |
| --- | --- |
| Acc. | correct / all questions |
| Ans. Rate | questions with an extracted option / all |
| Acc@Ans. | correct / answered |
| Grounded Acc. | correct and not "Cannot be determined" / all |

---

## 2. `scripts/qa_egomem.py` — EgoMem wrapper

Runs the EgoMemReason benchmark (`benchmarks/QA/egomem_reason_qa.json`, 500 questions over 6 wearers) through the `scripts/qa.py` pipeline **without editing it**. It reads `scripts/qa.py` from the same folder, applies asserted text patches, and executes the patched source as module `q_a_v6_egomem`.

### What it changes

| Concern | Handling |
| --- | --- |
| Options | A–D literals, regexes and prompt patched to A–J |
| Schema | `query_time "DAY7, 19:00:00"` → `{date, time}`; `options{}` → `choice_a…choice_j` |
| Types | `Activity Pattern`, `Spatial Preference` → `HabitInsight`; `Event Linking` → `EventRecall`; `Cumulative State Tracking`, `Temporal Counting` → LLM only; `Event Ordering` → own handler |
| Wearers | 6 identities (`A1_JAKE` … `A6_SHURE`); DB staged and camera wearer set per identity |
| Event Ordering | Per-event retrieval → timestamp → option with best pairwise concordance; LLM fallback with per-event timestamped evidence (`EGOMEM_ORDER_MODE`) |
| Labels | No `answer` field → predictions only; accuracy computed only when `answer` exists |

If a patch target is missing, it raises `[patch] … found 0x` — `scripts/qa.py` has drifted from the version the wrapper expects.

### Memory per person

Each wearer is answered from their own memory: `FIO_mem/<IDENTITY>_full` by default
(`--memory-template` / `EGOMEM_DB_TEMPLATE`, placeholders `{identity}`, `{wearer}`, `{build}`).
Override a single person with `--memory IDENTITY=PATH`, e.g. when a folder is named differently.
`--plan` shows, for every person, whether the memory exists and how many questions are done.

### Run

From `FIO/`:

```bash
python scripts/qa_egomem.py --plan                       # memory + progress per person
python scripts/qa_egomem.py --wearer jake                # one person
python scripts/qa_egomem.py --wearer jake,tasha          # several (names or ids, any case)
python scripts/qa_egomem.py --wearer all                 # everyone (the default)
python scripts/qa_egomem.py --wearer shure --memory A6_SHURE=../FIO_mem/A6_SHURE_RE_full
python scripts/qa_egomem.py --export                     # (re)build submission.json
```

People can be run in separate jobs (for example one per GPU) as long as they use the same
settings: every person writes their own file in the same run folder, and each is resumable.
A resume with different settings (memory, K, temperature, ranker, ...) is refused; use a new `--run-name`.
If a person is run with an override `--memory`, pass the same override when resuming them.

### Submitting to the EgoMemReason leaderboard

When every person is done, the run writes `submission.json` next to the predictions, in the
format the [EgoMemReason Space](https://huggingface.co/spaces/Ted412/EgoMemReason) scores:

```json
[{"example_id": 1, "predicted_answer": "A"}, ...]
```

It holds exactly one entry per question (all 500) and every letter is one of that question's
options, the checks the Space's `evaluator.py` applies. A question the pipeline could not answer
(no option, or a letter outside its options) is filled with its first option and counted in
`submission_report.json` (per person and question type), so check that report before uploading.
`--export --allow-missing` writes a submission before everyone is done, filling the missing people
the same way; use it only for testing the upload. Method name and team details are not part of the
file; the Space asks for them separately.

### Settings

| Option / variable | Default | Meaning |
| --- | --- | --- |
| `--wearer` / `EGOMEM_WEARER` | `all` | `all`, a name (`jake`), an id (`A1_JAKE`) or a comma list |
| `--memory-template` / `EGOMEM_DB_TEMPLATE` | `FIO_mem/{identity}_full` | memory folder per person |
| `--memory IDENTITY=PATH` | — | one person's memory folder (repeatable) |
| `EGOMEM_BUILD` | empty | value for `{build}` in the template |
| `--run-name` / `EGOMEM_RUN` | `<template>_<db_mode>_K<K>_T<temp>`, e.g. `full_opt_K5_T0` | run folder name |
| `EGOMEM_RESULTS_DIR` | `FIO/results/egomem` | parent of the run folders |
| `EGOMEM_QUESTIONS` | `benchmarks/QA/egomem_reason_qa.json` | question file |
| `EGOMEM_QTYPES` | all | comma list of `query_type` values (partial run: no submission) |
| `EGOMEM_LIMIT` | unset | max questions per person (debugging; partial run: no submission) |
| `EGOMEM_LLM_ONLY` | `Cumulative State Tracking,Temporal Counting` | types sent straight to the LLM |
| `EGOMEM_ORDER_MODE` | `hybrid` | `hybrid`, `heuristic` or `llm` |
| `EGOMEM_ORDER_MIN_SIM` | `0.40` | min similarity for an event to count as matched |
| `EGOMEM_ORDER_MIN_MATCHED` | `0.8` | matched fraction needed to use the heuristic in `hybrid` |
| `BUILD_CONFIG`, `DB_MODE`, `EVIDENCE_K`, `QA_TEMPERATURE`, `QA_RANKER`, `SEED`, `LOCAL_DB_ROOT` | as in `scripts/qa.py` | passed through |

### Output

```
results/egomem/<run>/
├── A1_JAKE.jsonl ... A6_SHURE.jsonl              one line per question (resumable)
├── A1_JAKE.config.json ...                       settings + environment per person
├── submission.json                               upload this
└── submission_report.json                        placeholders per person / type, run configs
```

Each prediction line:

```json
{"example_id": 112, "p_id": "A3_TASHA_...", "identity": "A3_TASHA", "query_type": "Event Ordering",
 "pred": "C", "path": "A", "grounded": true, "non_refusal": true,
 "answer_text": "Answer: C ...", "gold": null, "evidence_k": 5}
```

---

## Dependencies

Both scripts use `fio.llm` (Qwen, `call_qwen`), `fio.storage.dual` (`DualFIOStorage`), `fio.storage.staging` (`StagedChromaDB`), `fio.exp_logger` and `fio.repro`. Package versions are pinned in [`../requirements.txt`](../requirements.txt) (cluster conda env `egogpt`).

## Known issues

1. `extract_single_option_answer(answer, question)` can override the letter from the explanation text, but callers pass only `answer`, so the override never runs.
2. The β-weight ranking ablation (`q_a_beta_ablation.py`) is not in this repo; it used an older, α-weighted ranker.

