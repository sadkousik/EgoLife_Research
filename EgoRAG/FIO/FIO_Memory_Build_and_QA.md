# FIO — Memory Build and QA

This folder holds everything needed to **build** the FIO (fine-grained interaction object) memory from EgoGPT captions and to **answer questions** over it.

1. Qwen2.5-7B turns each caption into structured interaction records (FIOs).
2. `FIOOptimizer` filters, scores and merges them.
3. The results are stored in two ChromaDB stores:
   - `chroma_fio_raw` holds every FIO.
   - `chroma_fio_opt` holds the optimised FIOs.
4. `q_a_v6.py` retrieves from the opt store to answer EgoLifeQA questions.

```
FIO/
├── fio_v3.py                    # main build: stage ablations + hyperparameter sweeps
├── fio_v3_abl.py                # λ / τ / component ablation build (older optimizer)
├── q_a_v6.py                    # QA pipeline (see QA__Pipeline.md)
│
├── Q_decompose.py               # Qwen2.5-7B model, tokenizer, call_qwen
├── opti_fio_test.py             # FIOOptimizer used by fio_v3.py
├── opti_fio_abl.py              # FIOOptimizer with component switches, used by fio_v3_abl.py
├── prompt_fio_v2.py             # FIO extraction prompt (+ narrative prompts)
├── fio_schema.py                # FIO fields, validation, document string, scores
├── chroma_fio.py                # ChromaFIOStorage (low-level ChromaDB access)
├── chroma_fio_v2.py             # DualFIOStorage: raw + opt stores, search, date filter
├── fio_hierarchy.py             # slot / daily digests (only fio_v3_abl.py)
├── build_tracker.py             # BuildTracker → build_report.json
├── nfs_db_staging.py            # run ChromaDB on local disk, sync to NFS
├── experiment_logger.py         # start_experiment() run logging
├── fio_count.py                 # per-day entry counts of a Chroma store
│
├── fio_ablation.sh              # stage-ablation sweep (tmux)
├── fio_sweep.sh                 # one-hyperparameter sweep on the full model (tmux)
├── fio_mem_ablation.sh          # stage ablation on a day1..day1-6 cache window (tmux)
├── fio_mem_ablation_day1-7.sh   # same, on the full 7-day cache
├── fio_base1106.sh              # one full build in tmux
└── QA__Pipeline.md              # QA documentation
```

## Dependency map

```
fio_v3.py ─┬─ Q_decompose          (model, tokenizer)
           ├─ opti_fio_test        (FIOOptimizer)
           ├─ prompt_fio_v2        (FIO_EXTRACTION_PROMPT)
           ├─ fio_schema
           ├─ build_tracker
           ├─ chroma_fio_v2 ── chroma_fio ── fio_schema
           ├─ nfs_db_staging
           ├─ experiment_logger
           └─ fio_count

fio_v3_abl.py ── same, but opti_fio_abl instead of opti_fio_test, plus fio_hierarchy

q_a_v6.py ─┬─ Q_decompose
           ├─ chroma_fio_v2 ── chroma_fio ── fio_schema
           ├─ nfs_db_staging
           └─ experiment_logger
```

Python packages: `torch`, `transformers`, `chromadb`, `sentence-transformers` (`all-MiniLM-L6-v2`), `numpy`. Cluster conda env: `egogpt`.

## Where to run from

Run every script from the **`EgoRAG/` folder** (the parent of `FIO/`), calling the script by its path. Python puts `FIO/` on the import path automatically, so the imports between these files work. Relative data paths resolve against `EgoRAG/`: `memory_build_v2_7days/fio_cache`, `./BUILD_7day_ablation_*`, `./t_sne/` and `./ablation_runs/`.

```bash
cd EgoRAG
python -u FIO/fio_v3.py
```

The shell scripts `cd` into `PROJECT_DIR` (set at the top of each). Keep `PROJECT_DIR` pointing at `EgoRAG/` and call `FIO/fio_v3.py` inside them.

---

## 1. `fio_v3.py` — main build

### Two-phase pipeline

**Phase 1 — LLM extraction (GPU).**
- Each caption is prompted with `FIO_EXTRACTION_PROMPT`.
- `BATCH_SIZE` captions run through one `model.generate()` call, with left padding.
- Results are cached as one JSON file per caption: `{FIO_CACHE_DIR}/{date}-{start_time}.json`.
- It resumes from where it stopped, and a batch that runs out of GPU memory falls back to one caption at a time.
- Skipped by default (`SKIP_PHASE1 = True`), because the cache already exists.

**Phase 2 — optimisation (CPU, no LLM).** Runs in caption order, because the optimizer keeps state (prototypes, history window, per-object interactions):

1. Raw FIOs are stored in `chroma_fio_raw` (only with `STORE_RAW=1`).
2. `FIOOptimizer.process()` keeps, drops or merges FIOs. Merged history goes into `action_sequence`.
3. A per-day recency score is multiplied into `importance_score`.
4. The merged `action_sequence` is turned into text (`opt_document_extra`), and the kept FIOs go into `chroma_fio_opt`.

### FIO record

| Group | Fields |
| --- | --- |
| Identity | `id` (`{date}-{start_time}-{n}`), `date`, `start_time`, `end_time`, `video_path` |
| Interaction | `persons`, `action`, `object`, `tool`, `app`, `source_location`, `target_location`, `speech_content`, `attributes` (JSON string) |
| Scores | `heuristic_score`, `importance_score`, `recency_score`, `alpha_score` |
| Optimizer | `optimized`, `action_sequence` (JSON string), `opt_document_extra` |

### Stage ablations (`ABLATION`)

| Key | What changes | Optimizer override |
| --- | --- | --- |
| `full` | nothing (anchor) | — |
| `no_ctx` | contextual encoding off | `context_history_weight=0.0` |
| `no_s2` | importance pruning off (importance still computed) | `tau_threshold=-1.0` |
| `no_s3` | consolidation off | `compression_sim_threshold=2.0` |
| `no_s1` | prototype importance → uniform retention | `importance_mode="uniform"`, `target_keep_rate=0.4` |
| `s3_only` | only consolidation | `importance_mode="uniform"`, `target_keep_rate=1.0`, `compression_sim_threshold=0.75` |

Two of these modes need extra setup:
- `no_s1` and `s3_only` need a uniform-retention option in `opti_fio_test.py`. Without it, the script stops with `FIOOptimizer does not accept [...]`.
- For `no_s1`, set `target_keep_rate` to the full model's measured keep-rate (`total_kept / total_input` from its `build_report.json`). The 0.4 shown above is a placeholder.

### Base optimizer config (`BASE_OPT_KWARGS`)

| Knob | Value | Meaning |
| --- | --- | --- |
| `delta_g_threshold` | 0.75 | δ_g, prototype-assignment similarity |
| `tau_threshold` | 0.2 | τ, importance keep threshold |
| `delta_h_threshold` | 0.3 | δ_h, history similarity filter |
| `lambda_decay` | 1.146e-6 | λ, 7-day half-life |
| `context_history_weight` | 0.5 | β, history weight in the contextual embedding |
| `compression_sim_threshold` | 0.75 | Stage-3 merge gate |
| `history_window` | 5 | |
| `max_prototypes` | 300 | N_max |
| `max_object_interactions` | 20 | |
| `max_kept_for_compression` | 100 | |
| `redundancy_cooldown_sec` | 1.0 | |
| `track_growth` | True | |

`alpha_cold_start`, `alpha_ema`, `max_action_prototypes` and `max_alpha_clusters` are passed in but not used.

### Run

```bash
python -u FIO/fio_v3.py                                               # full build
ABLATION=no_s3 python -u FIO/fio_v3.py                                # one stage ablation
SWEEP_PARAM=delta_g_threshold SWEEP_VAL=0.65 python -u FIO/fio_v3.py  # one sweep point
STORE_RAW=1 python -u FIO/fio_v3.py                                   # also build the raw store (once)
```

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `ABLATION` | `full` | stage-ablation key |
| `SWEEP_PARAM`, `SWEEP_VAL` | unset | override one `BASE_OPT_KWARGS` knob (cast to the default's type) |
| `RUN_TAG` | `{ABLATION}` or `{ABLATION}__{SWEEP_PARAM}={SWEEP_VAL}` | names the output dir and t-SNE files |
| `OUTPUT_DIR` | `./BUILD_7day_ablation_{RUN_TAG}` | build output |
| `FIO_CACHE_DIR` | `memory_build_v2_7days/fio_cache` | Phase-1 cache |
| `TARGET_DAY` | `7` | use captions DAY1 … DAY`TARGET_DAY` |
| `STORE_RAW` | `False` | also build `chroma_fio_raw` (identical across ablations; build once) |
| `LOCAL_DB_ROOT` | unset | node-local staging dir; DBs synced back to NFS at the end |
| `SLURM_JOB_NAME`, `SLURM_JOB_ID` | `Build_7day_{ABLATION}` | experiment-log name |

### Set in the code (edit before running)

| Name | Value |
| --- | --- |
| `CAPTIONS_PATH` | `/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoGPT/meta_q_guided_7days_jake_05-02-2026-1.json` (Tasha path commented above it) |
| `SKIP_PHASE1` | `True`; set `False` to run LLM extraction |
| `TEST_DAY1_ONLY` | `False` |
| `BATCH_SIZE` | 6 (≈38 GB VRAM; 4 ≈ 29 GB) |
| `MAX_INPUT_TOKENS` | 3072 |

### Output

| Path | Content |
| --- | --- |
| `OUTPUT_DIR/chroma_fio_opt/` | optimised store (`q_a_v6.py` reads this with `DB_MODE=opt`) |
| `OUTPUT_DIR/chroma_fio_raw/` | raw store, only with `STORE_RAW=1` |
| `OUTPUT_DIR/build_report.json` | timings + `total_kept`, `total_dropped`, `total_compressed`, `proto_cap_hits` |
| `./t_sne/prototype_tsne_data_{RUN_TAG}.json` | prototype embeddings for t-SNE / UMAP |
| `./t_sne/merge_stats_{RUN_TAG}.json` | prototype assignment / merge counts |

At the end, the build prints per-day entry counts for each store (`fio_count.analyze_chroma_dates`).

---

## 2. Shell drivers for `fio_v3.py`

All drivers run Phase 2 only, so several builds can run at once on CPU. They share the same commands:

```bash
bash FIO/<script>.sh start | status | attach | stop | ls
```

Settings at the top of each script:
- `MAX_PARALLEL`: 0 = all at once, 1 = sequential, N = at most N running.
- `THREADS_PER_JOB`: keep (jobs × threads) at or below the physical core count.
- `CLEAN_FIRST=1`: wipe each run's output dir and t-SNE files before rebuilding.

Each run writes its own log, ending in a `___FINISHED___ … exit=N` line that `status` reads.

| Script | Runs | Output per run |
| --- | --- | --- |
| `fio_ablation.sh` | `ABLATIONS=(no_s2 no_ctx)` (edit the list) | `BUILD_7day_ablation_{name}/` |
| `fio_sweep.sh` | one `SWEEP_PARAM` over `SWEEP_VALUES` (uncomment one line: δ_g, N_max, τ, history, β, λ, δ_h) | `BUILD_7day_ablation_full__{param}={val}/` |
| `fio_mem_ablation.sh [window]` | `STAGES=(full no_s1 no_s2 no_s3 s3_only)` on `day1` … `day1-6` | `memory_build_v2_7days/fio_cache_{window}_mem/{stage}/` |
| `fio_mem_ablation_day1-7.sh` | same stages on the full cache | `memory_build_v2_7days/fio_cache_day1-7_mem/{stage}/` |
| `fio_base1106.sh` | one build with whatever is set in `fio_v3.py` (tmux session `imprint`) | `fio_v3.py` default |

`fio_mem_ablation.sh` needs the per-window caches made by `split_fio_cache_by_day.py` under `memory_build_v2_7days/fio_cache_windows/`.

---

## 3. `fio_v3_abl.py` — λ / τ / component ablation

An older build that reads λ and τ from the environment and can switch off individual terms of the importance score w(f). It still builds the hierarchy.

Use it only to reproduce the hyperparameter (Table 8) and component ablations; new experiments go through `fio_v3.py`. Its output is read by the β-ablation QA script.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ABLATION_TAG` | `Jake_7daysFinal` | run name; output in `./ablation_runs/{ABLATION_TAG}/` |
| `ABLATION_LAMBDA` | `8.0e-6` | λ (1 d = 8.0e-6, 3 d = 2.67e-6, 5 d = 1.60e-6, 7 d = 1.146e-6, 14 d = 5.73e-7) |
| `ABLATION_TAU` | `0.2` | τ keep threshold |
| `DISABLE_FREQUENCY`, `DISABLE_RECENCY`, `DISABLE_DISTINCTIVENESS`, `DISABLE_INTERACTION` | unset | `1` removes that term from w(f) |
| `CACHE_DIR` | `./memory_build_v2_7days` | Phase-1 cache root |

Fixed settings: `sim_threshold=0.75`, `compression_sim_threshold=0.75`, `history_window=5`, `max_prototypes=200`. The hierarchy uses `HierarchicalMemory(alpha_threshold=0.55, max_raw_per_slot=200)`.

```bash
ABLATION_TAG=abl_A6_lam5d_tau020_cpu ABLATION_LAMBDA=1.60e-6 ABLATION_TAU=0.20 \
CUDA_VISIBLE_DEVICES="" python3 FIO/fio_v3_abl.py
```

It writes these to `ablation_runs/{ABLATION_TAG}/`:
- `chroma_fio_raw/` and `chroma_fio_opt/`
- `hierarchy_memory.json`
- `build_report.json`
- `ablation_meta.json`, recording λ, τ, which components were on, `n_captions` and `n_kept`

---

## 4. `q_a_v6.py` — QA

`q_a_v6.py` answers EgoLifeQA questions by retrieving from a build's `chroma_fio_opt`. Its `BUILD_CONFIG` must name the same stage ablation as the build it reads.

```bash
DB_MODE=opt BUILD_CONFIG=full MAX_DAY=7 EVIDENCE_K=5 python -u FIO/q_a_v6.py
```

Full documentation is in [`QA__Pipeline.md`](QA__Pipeline.md).

---

## 5. Supporting modules

| Module | Provides |
| --- | --- |
| `Q_decompose.py` | Loads `Qwen/Qwen2.5-7B-Instruct` (bfloat16, `device_map="auto"`) at import; `call_qwen()`; the question-decomposition `system_message` |
| `fio_schema.py` | Field registry; `validate_fio`; `build_document_string`; `compute_heuristic_score` (field-presence score in [0, 1]); `compute_recency_for_day`; `parse_day_int` |
| `chroma_fio.py` | `ChromaFIOStorage`: persistent client, `m_store_fio_entry`, `semantic_search` |
| `chroma_fio_v2.py` | `DualFIOStorage` (raw + opt), `search` / `search_person_filter` / `search_speech_filter`, `get_where_filter` (DAY1…query day), `get_collection_name` (always `collection_day1_to_day7`) |
| `nfs_db_staging.py` | `StagedChromaDB`: `register()` + `sync_all()` for builds, `stage_in()` for read-only QA copies |
| `build_tracker.py` | `BuildTracker`: per-phase timing, token counts, ETA, `build_report.json` (stdlib only) |
| `fio_hierarchy.py` | `HierarchicalMemory`: slot digest every 120 clips (≈1 h), daily digest, `save()` → `hierarchy_memory.json` |
| `fio_count.py` | `analyze_chroma_dates(path)`: entries per collection and per `date` |
| `experiment_logger.py` | `start_experiment(name)`: run logger |

Importing `Q_decompose` loads the 7B model onto the GPU. Any script that imports it — including both builds, even with Phase 1 skipped — needs a GPU, or enough RAM to hold the model on CPU.

---

## Check that everything resolves

Run this inside `FIO/`. It parses each file and checks its imports without running anything:

```bash
python - <<'EOF'
import ast, glob, importlib.util, sys
sys.path.insert(0, ".")
missing = set()
for f in glob.glob("*.py"):
    for node in ast.walk(ast.parse(open(f).read(), f)):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for n in names:
            if importlib.util.find_spec(n.split(".")[0]) is None:
                missing.add((f, n))
for f, n in sorted(missing):
    print(f"{f}: cannot find '{n}'")
print("done" if not missing else f"{len(missing)} unresolved")
EOF
```

## Known issues

1. The opt-store embedding patch may have no effect. `fio_v3.py` swaps in its own document builder by reassigning `fio_schema.build_document_string`. But `chroma_fio_v2.py` imported that function by name when it loaded, so `store_opt_fios()` still calls the original. Merged `action_sequence` text may therefore never reach the embeddings. To fix it, set `f["document"] = build_document_string_opt_aware(f)` on each kept FIO before `store_opt_fios()`. This changes the embeddings, so existing opt builds would need rebuilding.
2. `no_s1` and `s3_only` need the uniform-retention patch in `opti_fio_test.py`.
3. `q_a_v6.py` has no ranker setting for `no_ctx`, so `BUILD_CONFIG=no_ctx` raises `ValueError`.
4. `fio_v3_abl.py` uses the older optimizer (`max_prototypes=200`, α still in the importance score, hierarchy on). Its builds are not directly comparable with `fio_v3.py` builds.
5. `CAPTIONS_PATH`, `SKIP_PHASE1` and the shell scripts' `PROJECT_DIR` are hard-coded to the cluster path `/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG`. Edit them to run elsewhere.
6. `fio_mem_ablation_day1-7.sh` duplicates `fio_mem_ablation.sh` except for the cache path.