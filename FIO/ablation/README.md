# Ablations

Everything for the memory-build ablation studies of the paper
(Imprint, arXiv:2607.00696) lives here.

| File | Role |
| --- | --- |
| `configs.py` | The locked base config (`BASE_OPT_KWARGS`), every ablation row (`ABLATIONS`) and the sweep logic. `scripts/build_memory.py` reads it. |
| `run.sh` | Launcher: builds many variants in parallel in tmux (Phase 2 only, CPU), then evaluates them (`qa`) and tabulates GA / Acc. (`report`). |

List the rows and what each one changes:

```bash
python ablation/configs.py
```

## Paper notation → code

| Paper | Code setting | Value |
| --- | --- | --- |
| δ_g, prototype-assignment threshold | `delta_g_threshold` | 0.75 |
| N_max, prototype cap | `max_prototypes` | 300 |
| k, history length | `history_window` | 5 |
| α_h, context-history weight | `context_history_weight` | 0.5 |
| δ_h, history-similarity threshold | `delta_h_threshold` | 0.3 |
| λ, recency decay | `lambda_decay` (per **second**) | 1.146e-6 (paper writes 1.14, i.e. ×10⁻⁶ s⁻¹) |
| τ, importance threshold | `tau_threshold` | 0.2 |
| δ_m, consolidation threshold | `compression_sim_threshold` | 0.75 |
| γ₁, γ₂, γ₃ (Eq. 3) | `qa.py` ranker `full` (`QA_RANKER`) | 0.53, 0.29, 0.18 |
| Top-K evidence | `EVIDENCE_K` | 5 |
| θ, match threshold (Alg. 2) | `min_score` in `qa.py` handlers | 0.35 |
| GA, grounded accuracy | `fio/grounding.py` (`metrics.GA` in the QA results) | grounded correct / correct |

Not in the paper but part of the system: before Eq. 3 ranks candidates, a Chroma
search returns a candidate pool re-scored by `0.55·semantic + 0.20·recency + 0.20·importance`
and keeps the top 30 (`fio/storage/chroma.py`). QA decoding is greedy by default
(`QA_TEMPERATURE=0`); Phase-1 extraction samples at temperature 0.9 (seeded by `SEED`).

## Rows (`ABLATION=<name>`)

Each row starts from `BASE_OPT_KWARGS` and changes one factor.

**Stages (Table 2, Table 5)**

| Name | Paper row | Override |
| --- | --- | --- |
| `full` | Imprint | — |
| `no_s1` | w/o Event-Prototype Assignment (uniform retention at the full model's keep-rate) | `importance_mode="uniform"`, `target_keep_rate=0.4` |
| `no_s2` | w/o Interaction Importance (keep all; score still computed) | `tau_threshold=-1.0` |
| `no_s3` | w/o Interaction Consolidation (never merge) | `compression_sim_threshold=2.0` |
| `s3_only` | Interaction Consolidation only | `importance_mode="uniform"`, `target_keep_rate=1.0` |
| `no_ctx` | (α_h = 0; Fig. 7c end point) | `context_history_weight=0.0` |

For `no_s1`, set `target_keep_rate` in `configs.py` to the full model's measured
`total_kept / total_input` (from its `build_report.json`). 0.4 is a placeholder.

**Importance signals (Table 3): terms of w(f) = log(1+ν) · e^(−λΔT) · 1/(1 + log(1+N))**

| Name | Term set to 1 | Override |
| --- | --- | --- |
| `no_freq` | frequency, log(1+ν) | `disable_frequency=True` |
| `no_recency` | recency, e^(−λΔT); same as λ = 0 | `disable_recency=True` |
| `no_distinct` | distinctiveness, 1/(1+log(1+N)) | `disable_distinctiveness=True` |

τ stays at its base value, so removing a term also shifts how many records are kept;
compare `total_kept` in each run's `build_report.json` alongside GA / Acc.

## Sweeps (`SWEEP_PARAM` / `SWEEP_VAL`)

A sweep changes one `BASE_OPT_KWARGS` knob on top of any row (the value is cast to the default's type).

| Knob | Paper grid | Where |
| --- | --- | --- |
| `max_prototypes` | 100 200 300 400 500 | Fig. 6a |
| `delta_g_threshold` | 0.60 0.65 0.70 0.75 0.80 0.85 | Fig. 6b |
| `lambda_decay` | 2.6742e-6 1.6045e-6 1.146e-6 5.73e-7 | Fig. 7a (paper: 2.67, 1.60, 1.14, 0.57) |
| `tau_threshold` | 0.1 0.2 0.3 | Fig. 7b |
| `context_history_weight` | 0.3 0.5 0.7 | Fig. 7c |
| `history_window` | 3 5 10 | Table 6 |
| `delta_h_threshold` | 0.2 0.3 0.5 0.7 | Table 7 |

## Reproducing the paper's tables

Run from `FIO/`. Builds are CPU-only and run in tmux; `qa` needs the GPU, so run it
inside a Slurm job (e.g. replace the commands at the end of `jobs/qw.sh`). Defaults are
the paper's setting: `WEARER=A1_JAKE`, the 500 EgoLifeQA questions, `SEED=1234`.
Each `qa` writes `results/ablation/<session>_<tag>.json` (config, GA, Acc., every
prediction, environment) and skips builds that already have results; `report` prints the table.

```bash
# Table 1 (Imprint) + Table 2: stage ablation
STAGES="full no_s1 no_s2 no_s3 s3_only" ablation/run.sh stages start      # then: ... stages status
STAGES="full no_s1 no_s2 no_s3 s3_only" ablation/run.sh stages qa         # on a GPU node

# Table 3: importance signals (reuses the full build above if it exists)
STAGES="full no_freq no_recency no_distinct" ablation/run.sh stages start
STAGES="full no_freq no_recency no_distinct" ablation/run.sh stages qa

# Table 5: stage ablation per cumulative horizon (W = day1, day1-2, ..., day1-7)
for W in day1 day1-2 day1-3 day1-4 day1-5 day1-6 day1-7; do
  STAGES="full no_s1 no_s2 no_s3 s3_only" ablation/run.sh window $W start; done
#   (wait for the builds; run.sh window <W> status)
for W in day1 day1-2 day1-3 day1-4 day1-5 day1-6 day1-7; do
  STAGES="full no_s1 no_s2 no_s3 s3_only" ablation/run.sh window $W qa; done

# Fig. 6, Fig. 7, Tables 6-7: one sweep per knob (grids above), e.g. Table 7
SWEEP_PARAM=delta_h_threshold SWEEP_VALUES="0.2 0.3 0.5 0.7" ablation/run.sh sweep start
SWEEP_PARAM=delta_h_threshold SWEEP_VALUES="0.2 0.3 0.5 0.7" ablation/run.sh sweep qa
#   one sweep session at a time: they share the tmux session name fio_abl_sweep

# Table 8 (retrieval-score terms) and Table 9 (Top-K): QA only, on the full build
B=../FIO_mem/A1_JAKE_ablation/full
for R in no_e no_a no_imp; do QA_BUILD_DIR=$B QA_RANKER=$R QA_RESULTS=results/ablation/table8_$R.json python -u scripts/qa.py; done
for K in 2 5 10 15 20;     do QA_BUILD_DIR=$B EVIDENCE_K=$K QA_RESULTS=results/ablation/table9_K$K.json python -u scripts/qa.py; done
```

`report` reads the same settings as the matching `start`, so call it with the same
`STAGES` / `SWEEP_*` / window, e.g. `STAGES="full no_freq no_recency no_distinct" ablation/run.sh stages report`.

## One run

```bash
ABLATION=no_s3 python -u scripts/build_memory.py
SWEEP_PARAM=tau_threshold SWEEP_VAL=0.3 python -u scripts/build_memory.py
ABLATION=no_recency SKIP_PHASE1=1 OUTPUT_DIR=../FIO_mem/test python -u scripts/build_memory.py
```

The run tag is `<row>` or `<row>__<knob>=<value>` (override it with `RUN_TAG`). It names
the output dir and the `t_sne/` exports, so runs never overwrite each other.

## `run.sh` reference

```bash
ablation/run.sh stages     plan|start|status|attach|stop|qa|report   # every row in STAGES
ablation/run.sh sweep      plan|start|status|attach|stop|qa|report   # SWEEP_PARAM over SWEEP_VALUES
ablation/run.sh window W   plan|start|status|attach|stop|qa|report   # STAGES on days 1..N
ablation/run.sh ls                                                   # tmux sessions
```

Every setting below can be set from the environment.

| Setting | Default | Meaning |
| --- | --- | --- |
| `WEARER` | `A1_JAKE` | picks the captions, cache and output paths below |
| `STAGES` | `no_ctx no_s1 no_s2 no_s3 s3_only` | rows for `stages` and `window` |
| `SWEEP_BASE`, `SWEEP_PARAM`, `SWEEP_VALUES` | `full`, `delta_h_threshold`, `0.2 0.3 0.5 0.7` | sweep |
| `MAX_PARALLEL` | `0` | 0 = all at once, 1 = sequential, N = at most N |
| `THREADS_PER_JOB` | `4` | keep (jobs × threads) ≤ physical cores |
| `CLEAN_FIRST` | `0` | 1 = delete a run's output dir and t-SNE files before rebuilding |
| `CAPTIONS_PATH` | `generated_caption/<WEARER>_meta.json` | |
| `FIO_CACHE_DIR` | `FIO_mem/<WEARER>/fio_cache` | Phase-1 cache (must exist; Phase 1 is skipped). Serves every day window. |
| `OUT_ROOT` | `FIO_mem/<WEARER>_ablation` | builds: `<tag>/`, or `<W>/<row>/` for `window` |
| `LOG_DIR` | `FIO/log/ablation` | build and QA logs |
| `QA_DATA_PATH` | `benchmarks/QA/EgoLifeQA_A1_JAKE.json` | questions for `qa` |
| `RESULTS_DIR` | `FIO/results/ablation` | QA results for `qa` / `report` |

`stop` kills only the builds this launcher started.
