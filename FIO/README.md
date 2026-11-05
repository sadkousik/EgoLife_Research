# FIO — Memory Build and QA

Builds the FIO (fine-grained interaction object) memory from EgoGPT captions and answers questions over it.

```
FIO/
├── fio/                    # importable library (no scripts here)
│   ├── llm.py              # Qwen2.5-7B backend: call_qwen; model loads on first use, pinned revision
│   ├── prompts.py          # FIO extraction prompt
│   ├── schema.py           # FIO fields, validation, document strings, recency
│   ├── optimizer.py        # FIOOptimizer: filter, score, merge
│   ├── tracking.py         # BuildTracker -> build_report.json
│   ├── exp_logger.py       # start_experiment() run logging
│   ├── stats.py            # per-day entry counts of a Chroma store
│   ├── repro.py            # set_seed(), run_info() (git, versions, model snapshots)
│   ├── grounding.py        # Grounded Accuracy (paper Sec. 4.1): answer path + self-admission rule
│   └── storage/
│       ├── chroma.py       # ChromaFIOStorage (one store)
│       ├── dual.py         # DualFIOStorage: raw + opt stores, search, filters
│       └── staging.py      # StagedChromaDB: node-local Chroma, synced back to NFS
├── scripts/                # entry points
│   ├── build_memory.py     # build the memory (stage ablations, sweeps)
│   ├── qa.py               # EgoLifeQA pipeline
│   └── qa_egomem.py        # EgoMemReason: per-person memory, leaderboard submission.json
├── ablation/               # ablation rows, base config, sweeps + tmux launcher (run.sh)
├── jobs/                   # single runs: qw.sh (Slurm), build_tmux.sh (tmux)
├── analysis/               # hallucination / grounding analysis scripts
├── docs/                   # memory_build_and_qa.md, qa_pipeline.md
├── requirements.txt        # pinned package versions
├── results/                # QA results + EgoMem predictions (git-ignored)
├── experiments/            # per-run experiment logs (git-ignored)
├── log/                    # Slurm / launcher logs (git-ignored)
└── t_sne/                  # t-SNE / merge-stat exports (git-ignored)
```

## Run

Run from this folder, so outputs land in `log/` and `t_sne/` here:

```bash
cd EgoLife_Research/FIO
python -u scripts/build_memory.py                 # build memory
python scripts/qa_egomem.py --wearer all          # EgoMemReason QA → results/egomem/<run>/submission.json
sbatch jobs/qw.sh                                 # both, on Slurm
ablation/run.sh stages plan                       # ablation runs: see ablation/README.md
```

The scripts set up their own import path, so no install is needed. Inside code, import from the package, for example `from fio.optimizer import FIOOptimizer`. Outside this folder (as `learnable/` does), put `FIO/` on `sys.path` first.

## Reproducing a result

1. Install the pinned versions: `pip install -r requirements.txt` (the reported runs used the cluster's `egogpt` env, Python 3.10).
2. Build: every build writes `run_config.json` into its output folder (ablation row, final optimizer settings, seed, inputs, LLM revision, git commit, package versions). To rerun one, set the same environment variables it lists.
3. Evaluate: `QA_BUILD_DIR=<build> python -u scripts/qa.py`. QA decodes greedily (`QA_TEMPERATURE=0`) and writes `results/qa_*.json` with the config, metrics, every prediction and the environment.
4. Keep the `fio_cache/` with the results. Phase 1 (LLM extraction) samples, so the cache is what makes Phase 2 and QA repeatable.

Everything random is seeded from `SEED` (default 1234). Runs from before these changes sampled during QA (temperature 0.6, no seed), so their numbers can differ by run; `QA_TEMPERATURE=0.6` restores that decoding.

Job-finished emails are sent only if `EXP_NOTIFY_EMAIL` and `EXP_NOTIFY_APP_PASSWORD` are set in the environment. Never put credentials in the code.

Settings and environment variables for each script are in [docs/memory_build_and_qa.md](docs/memory_build_and_qa.md) and [docs/qa_pipeline.md](docs/qa_pipeline.md).

## Old file names

| Old | New |
| --- | --- |
| `Q_decompose.py` | `fio/llm.py` |
| `prompt_fio_v2.py` | `fio/prompts.py` |
| `fio_schema.py` | `fio/schema.py` |
| `opti_fio_test.py` | `fio/optimizer.py` |
| `fio_hierarchy.py` | removed (the slot / daily digest layer was unused) |
| `build_tracker.py` | `fio/tracking.py` |
| `experiment_logger.py` | `fio/exp_logger.py` |
| `fio_count.py` | `fio/stats.py` |
| `chroma_fio.py` | `fio/storage/chroma.py` |
| `chroma_fio_v2.py` | `fio/storage/dual.py` |
| `nfs_db_staging.py` | `fio/storage/staging.py` |
| `fio_v3.py` | `scripts/build_memory.py` |
| `q_a_v6.py` | `scripts/qa.py` |
| `q_a_egomem.py` | `scripts/qa_egomem.py` |
| `Grounded_result/` | `analysis/` |
| `fio_ablation.sh`, `fio_sweep.sh`, `fio_mem_ablation*.sh` | `ablation/run.sh` (stages / sweep / window) |
| `fio_base1106.sh` | `jobs/build_tmux.sh` |
| `fio_v3_abl.py`, `opti_fio_abl.py` | removed; their λ/τ sweeps and term ablations are rows in `ablation/configs.py` |
