# EgoRAG — QA Pipeline

Question answering over first-person video memory built as **FIO** (fine-grained interaction object) records in ChromaDB. This folder holds one canonical QA pipeline and two thin drivers on top of it.

```
EgoRAG/
├── q_a_v6.py             # canonical pipeline (from q_a_v62706.py)
├── q_a_egomem.py         # EgoMem benchmark wrapper, loads q_a_v6.py at runtime
└── q_a_beta_ablation.py  # β-weight ranking ablation (was q_a_beta.py)
```

| File | Benchmark | Ranker | Called by |
| --- | --- | --- | --- |
| `q_a_v6.py` | EgoLifeQA (A–D options) | 3-term: entity, action, importance | `qa_mem_ablation.sh`, `qw.sh` |
| `q_a_egomem.py` | EgoMem (A–J options, 6 wearers) | same as `q_a_v6.py` | `qw.sh` |
| `q_a_beta_ablation.py` | EgoLifeQA | 4-term: entity, action, importance, α | `beta_ablation.sh`, `C_eval.sh` |

---

## 1. `q_a_v6.py` — canonical pipeline

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

   `alpha_score` is **not** used in ranking (logged only).

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

A hierarchy cascade (slot / daily digests) exists in code but is **off** (`USE_HIERARCHY = False`).

### Run

```bash
DB_MODE=opt BUILD_CONFIG=full MAX_DAY=7 EVIDENCE_K=5 python -u q_a_v6.py
```

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `EVIDENCE_K` | `5` | Max evidence items per LLM call (evidence-count ablation) |
| `DB_MODE` | `opt` | `opt` = merged FIO DB, `raw` = unmerged baseline |
| `BUILD_CONFIG` | `full` | Memory build variant → ranker: `full`, `no_s3` → `full`; `no_s1`, `no_s2`, `s3_only` → `no_imp` |
| `MEM_WINDOW` | `day1-7` | Label used in the experiment name |
| `MAX_DAY` | `7` | Evaluate questions from DAY1 … DAY`MAX_DAY` |
| `LOCAL_DB_ROOT` | unset | Node-local path to stage the Chroma DB into (`nfs_db_staging.StagedChromaDB`) |

Evidence-K sweep:

```bash
for K in 2 5 10 15 20; do EVIDENCE_K=$K DB_MODE=opt python -u q_a_v6.py; done
```

### Hard-coded paths in `__main__` (edit before running)

| Name | Current value |
| --- | --- |
| `ab_folder_name` | `BUILD_7day_ablation_full__delta_h_threshold=0.7` |
| `DATA_PATH` | `/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoLifeQA_A1_JAKE.json` |
| `camera_wearer` | `Jake` |
| DB root | `/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/{ab_folder_name}/chroma_fio_{opt,raw}` |

> `qa_mem_ablation.sh` sets `AB_FOLDER_NAME` and `DATA_PATH`, but `q_a_v6.py` does not read them yet. Until it does, every stage evaluates the hard-coded build.

### Output

Per-type and global metrics printed at the end, plus the experiment log from `experiment_logger.start_experiment("{BUILD_CONFIG}_{MEM_WINDOW}_K{EVIDENCE_K}")`.

| Metric | Definition |
| --- | --- |
| Acc. | correct / all questions |
| Ans. Rate | questions with an extracted option / all |
| Acc@Ans. | correct / answered |
| Grounded Acc. | correct and not "Cannot be determined" / all |

---

## 2. `q_a_egomem.py` — EgoMem wrapper

Runs the EgoMem benchmark (`egomem_reason_q.json`) through the `q_a_v6.py` pipeline **without editing it**. It reads `q_a_v6.py` from the same folder, applies asserted text patches, and executes the patched source as module `q_a_v6_egomem`.

### What it changes

| Concern | Handling |
| --- | --- |
| Options | A–D literals, regexes and prompt patched to A–J |
| Schema | `query_time "DAY7, 19:00:00"` → `{date, time}`; `options{}` → `choice_a…choice_j` |
| Types | `Activity Pattern`, `Spatial Preference` → `HabitInsight`; `Event Linking` → `EventRecall`; `Cumulative State Tracking`, `Temporal Counting` → LLM only; `Event Ordering` → own handler |
| Wearers | 6 identities (`A1_JAKE` … `A6_SHURE`); DB staged and camera wearer set per identity |
| Event Ordering | Per-event retrieval → timestamp → option with best pairwise concordance; LLM fallback with per-event timestamped evidence (`EGOMEM_ORDER_MODE`) |
| Labels | No `answer` field → predictions only; accuracy computed only when `answer` exists |

If a patch target is missing, it raises `[patch] … found 0x` — `q_a_v6.py` has drifted from the version the wrapper expects.

### Run

```bash
python q_a_egomem.py --wearer tasha
EGOMEM_BUILD=build_mem/BUILD_7day_Tasha_full EVIDENCE_K=5 DB_MODE=opt python q_a_egomem.py --wearer A3_TASHA
```

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `EGOMEM_QUESTIONS` | `/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/egomem_reason_q.json` | Question file |
| `EGOMEM_OUT` | `egomem_preds_tasha_K{EVIDENCE_K}.jsonl` | Predictions (resumable: done `p_id`s are skipped) |
| `EGOMEM_BUILD` | `build_mem/BUILD_7day_Tasha_full` | Build folder name |
| `EGOMEM_DB_TEMPLATE` | `/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/{build}` | DB root; placeholders `{build}`, `{identity}`, `{wearer}` |
| `EGOMEM_WEARER` | `all` | `all`, a name (`jake`), an id (`A1_JAKE`) or a comma list; `--wearer` overrides |
| `EGOMEM_QTYPES` | all | Comma list of `query_type` values |
| `EGOMEM_LLM_ONLY` | `Cumulative State Tracking,Temporal Counting` | Types sent straight to the LLM |
| `EGOMEM_ORDER_MODE` | `hybrid` | `hybrid`, `heuristic` or `llm` |
| `EGOMEM_ORDER_MIN_SIM` | `0.40` | Min similarity for an event to count as matched |
| `EGOMEM_ORDER_MIN_MATCHED` | `0.8` | Matched fraction needed to use the heuristic in `hybrid` |
| `EGOMEM_LIMIT` | unset | Max questions (debugging) |
| `BUILD_CONFIG`, `DB_MODE`, `EVIDENCE_K`, `LOCAL_DB_ROOT` | as in `q_a_v6.py` | Passed through |

### Output

One JSON line per question in `EGOMEM_OUT`:

```json
{"p_id": "...", "identity": "A3_TASHA", "query_type": "Event Ordering",
 "pred": "C", "grounded": true, "answer_text": "Answer: C ...", "gold": null, "evidence_k": 5}
```

---

## 3. `q_a_beta_ablation.py` — β-weight ablation

Env-driven evaluation for the ranking-weight ablation. Its handler code is the **earlier** generation (before temporal candidate pool, calendar-aware WHEN, qualifier-aware WHO and conditioned co-occurrence), and its ranker includes α:

```
opt: score = β1·entity + β2·action + β3·importance + β4·alpha     (default 0.45, 0.25, 0.15, 0.15)
raw: score = 0.50·entity + 0.30·action + 0.20·importance
```

`EGOLIFE_ABL_BETAS` overrides all four weights in both modes. Results from this script are **not** directly comparable with `q_a_v6.py` numbers.

### Variants (`beta_ablation.sh`, Slurm array 0–4)

| Index | `BETA_TAG` | `EGOLIFE_ABL_BETAS` |
| --- | --- | --- |
| 0 | `full` | `0.45,0.25,0.15,0.15` |
| 1 | `no_entity` | `0.0,0.4545,0.2727,0.2727` |
| 2 | `no_action` | `0.6,0.0,0.2,0.2` |
| 3 | `no_w` | `0.5294,0.2941,0.0,0.1765` |
| 4 | `no_kappa` | `0.5294,0.2941,0.1765,0.0` |

```bash
sbatch beta_ablation.sh                 # all 5 variants
bash beta_ablation.sh aggregate         # LaTeX table via aggregate_beta_results.py
```

Single run:

```bash
ABLATION_TAG=abl_A3_lam7d_tau020_cpu DB_MODE=opt MAX_DAY=7 \
EGOLIFE_ABL_BETAS=0.45,0.25,0.15,0.15 BETA_TAG=full \
RESULTS_JSON=beta_ablation_results/raw/full.json \
python -u q_a_beta_ablation.py
```

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `ABLATION_TAG` | `Jake_7daysFinal` | Build: default → `./memory_build_Jake_7daysFinal`, otherwise `./ablation_runs/{ABLATION_TAG}` |
| `DATA_PATH` | `.../EgoLifeQA_A1_JAKE.json` | Question file |
| `DB_MODE` | **`raw`** | Set `opt` explicitly for ablations |
| `MAX_DAY` | `7` | Day window |
| `EGOLIFE_ABL_BETAS` | unset | `b1,b2,b3,b4` ranking weights |
| `BETA_TAG` | `default` | Variant name written to results |
| `RESULTS_JSON` | unset | If set, writes results JSON |

### Output (`RESULTS_JSON`)

```json
{
  "ablation_tag": "...", "beta_tag": "full", "betas": "0.45,0.25,0.15,0.15",
  "db_mode": "opt", "build_meta": { "...from ablation_meta.json..." },
  "per_type": { "DAY7": { "EntityLog": { "accuracy": 0.0, "overall_accuracy": 0.0, "...": 0 } } },
  "global": { "accuracy": 0.0, "answer_rate": 0.0, "acc_at_ans": 0.0, "grounded_acc": 0.0,
              "total": 0, "correct": 0, "answered": 0, "grounded": 0 }
}
```

---

## Dependencies

All three scripts import these local modules and packages:

| Module | Used for |
| --- | --- |
| `Q_decompose` | Qwen model, tokenizer, `call_qwen`, `system_message` |
| `chroma_fio_v2` | `DualFIOStorage`, `get_collection_name` |
| `experiment_logger` | `start_experiment` |
| `nfs_db_staging` | `StagedChromaDB` (`q_a_v6.py`, `q_a_egomem.py` only) |
| `chromadb`, `sentence-transformers` | `all-MiniLM-L6-v2` embeddings |
| `numpy` | similarity scoring |

Conda environment on the cluster: `egogpt`.

## Known issues

1. `q_a_v6.py` ignores `AB_FOLDER_NAME`, `DATA_PATH` and camera wearer from the environment (hard-coded in `__main__`).
2. `extract_single_option_answer(answer, question)` can override the letter from the explanation text, but callers pass only `answer`, so the override never runs.
3. `q_a_beta_ablation.py` defaults to `DB_MODE=raw`; ablation scripts must set `opt`.
4. The β ablation uses the α-weighted ranker and older handlers; report it as an ablation of that ranker, or port the `EGOLIFE_ABL_BETAS` hook into `q_a_v6.py` and re-run.

Older QA versions (`q_a_v2` … `q_a_v6_abl`) are kept in `archive/qa_history/` for reproducibility only.