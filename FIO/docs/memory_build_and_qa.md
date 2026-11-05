# FIO — Memory Build and QA

This folder holds everything needed to **build** the FIO (fine-grained interaction object) memory from EgoGPT captions and to **answer questions** over it.

1. Qwen2.5-7B turns each caption into structured interaction records (FIOs).
2. `FIOOptimizer` filters, scores and merges them.
3. The results are stored in two ChromaDB stores:
   - `chroma_fio_raw` holds every FIO.
   - `chroma_fio_opt` holds the optimised FIOs.
4. `scripts/qa.py` retrieves from the opt store to answer EgoLifeQA questions.

The folder layout is described in [`../README.md`](../README.md).

## Dependency map

```
scripts/build_memory.py ─┬─ fio.llm             (model, tokenizer, call_qwen)
                         ├─ fio.optimizer       (FIOOptimizer)
                         ├─ ablation.configs    (base config, rows, sweeps)
                         ├─ fio.prompts         (FIO_EXTRACTION_PROMPT)
                         ├─ fio.schema
                         ├─ fio.tracking        (BuildTracker)
                         ├─ fio.storage.dual ── fio.storage.chroma ── fio.schema
                         ├─ fio.storage.staging (StagedChromaDB)
                         ├─ fio.exp_logger
                         └─ fio.stats

scripts/qa.py ─┬─ fio.llm
               ├─ fio.storage.dual ── fio.storage.chroma ── fio.schema
               ├─ fio.storage.staging
               └─ fio.exp_logger

scripts/qa_egomem.py ── loads scripts/qa.py's source at runtime (patched for A–J options)
```

Python packages are pinned in [`../requirements.txt`](../requirements.txt) (cluster conda env `egogpt`, Python 3.10).

## Where to run from

Each script under `scripts/` puts the `FIO/` folder on the import path itself, so `import fio` works from any working directory. Relative data paths (`./t_sne/`, `log/`, `BUILD_*`, `memory_build_v2_7days/fio_cache`, output `.jsonl` files) resolve against the folder you run from, so run from `FIO/` to keep outputs in `FIO/t_sne/` and `FIO/log/`:

```bash
cd EgoLife_Research/FIO
python -u scripts/build_memory.py
sbatch jobs/qw.sh            # Slurm: build memory, then EgoMem QA
```

For many builds at once (ablations, sweeps, day windows) use `ablation/run.sh`; for one build in tmux, `jobs/build_tmux.sh`.

---

## 1. `scripts/build_memory.py` — main build

### Two-phase pipeline

**Phase 1 — LLM extraction (GPU).**
- Each caption is prompted with `FIO_EXTRACTION_PROMPT`.
- `BATCH_SIZE` captions run through one `model.generate()` call, with left padding.
- Results are cached as one JSON file per caption: `{FIO_CACHE_DIR}/{date}-{start_time}.json`.
- It resumes from where it stopped, and a batch that runs out of GPU memory falls back to one caption at a time.
- Runs by default and resumes: captions already in the cache are skipped. `SKIP_PHASE1=1` skips the phase entirely.

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
| Scores | `heuristic_score`, `importance_score`, `recency_score` |
| Optimizer | `optimized`, `action_sequence` (JSON string), `opt_document_extra` |

### Ablations and sweeps

`ABLATION`, `SWEEP_PARAM` and `SWEEP_VAL` select a row and an optional one-knob sweep. The rows (stages and w(f) terms), the locked base config and the tmux launcher are in [`ablation/`](../ablation/README.md).

### Run

```bash
python -u scripts/build_memory.py                                               # full build
ABLATION=no_s3 python -u scripts/build_memory.py                                # one ablation row
SWEEP_PARAM=delta_g_threshold SWEEP_VAL=0.65 python -u scripts/build_memory.py  # one sweep point
STORE_RAW=1 python -u scripts/build_memory.py                                   # also build the raw store (once)
```

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `ABLATION` | `full` | ablation row (see `ablation/configs.py`) |
| `SWEEP_PARAM`, `SWEEP_VAL` | unset | override one `BASE_OPT_KWARGS` knob (cast to the default's type) |
| `RUN_TAG` | `{ABLATION}` or `{ABLATION}__{SWEEP_PARAM}={SWEEP_VAL}` | names the output dir and t-SNE files |
| `CAPTIONS_PATH` | `…/EgoLife_Research/generated_caption/A6_SHURE_meta.json` | EgoGPT captions |
| `OUTPUT_DIR` | `…/EgoLife_Research/FIO_mem/A6_SHURE_RE_{RUN_TAG}` | build output |
| `FIO_CACHE_DIR` | `…/EgoLife_Research/FIO_mem/A6_SHURE/fio_cache` | Phase-1 cache |
| `SKIP_PHASE1` | `0` | `1` = use the existing cache, no LLM extraction (Phase 1 also resumes, skipping cached captions) |
| `TARGET_DAY` | `7` | use captions DAY1 … DAY`TARGET_DAY` |
| `SEED` | `1234` | seeds Python, NumPy and torch, including Phase-1 sampling |
| `QWEN_MODEL_ID`, `QWEN_REVISION` | `Qwen/Qwen2.5-7B-Instruct`, `a09a354…` | LLM and its pinned Hugging Face snapshot |
| `STORE_RAW` | `False` | also build `chroma_fio_raw` (identical across ablations; build once) |
| `LOCAL_DB_ROOT` | unset | node-local staging dir; DBs synced back to NFS at the end |
| `SLURM_JOB_NAME`, `SLURM_JOB_ID` | `Build_7day_{ABLATION}` | experiment-log name |

### Set in the code (edit before running)

| Name | Value |
| --- | --- |
| `TEST_DAY1_ONLY` | `False` |
| `BATCH_SIZE` | 6 (≈38 GB VRAM; 4 ≈ 29 GB) |
| `MAX_INPUT_TOKENS` | 3072 |

### Output

| Path | Content |
| --- | --- |
| `OUTPUT_DIR/chroma_fio_opt/` | optimised store (`scripts/qa.py` reads this with `DB_MODE=opt`) |
| `OUTPUT_DIR/chroma_fio_raw/` | raw store, only with `STORE_RAW=1` |
| `OUTPUT_DIR/run_config.json` | everything needed to rerun: row, final optimizer kwargs, seed, input paths, LLM revision, git commit, package versions; `status` becomes `finished` with the kept count at the end |
| `OUTPUT_DIR/build_report.json` | timings + `total_kept`, `total_dropped`, `total_compressed`, `proto_cap_hits` |
| `./t_sne/prototype_tsne_data_{RUN_TAG}.json` | prototype embeddings for t-SNE / UMAP |
| `./t_sne/merge_stats_{RUN_TAG}.json` | prototype assignment / merge counts |

At the end, the build prints per-day entry counts for each store (`fio.stats.analyze_chroma_dates`).

---

## 2. Launchers

| Launcher | Runs |
| --- | --- |
| `ablation/run.sh` | many ablation / sweep / day-window builds in tmux. See [`ablation/README.md`](../ablation/README.md). |
| `jobs/qw.sh` | Slurm: one build, then EgoMem QA. Submit from `FIO/`: `sbatch jobs/qw.sh`. |
| `jobs/build_tmux.sh` | one build in tmux session `fio_build`; passes the build's environment variables into the session. |

---

## 3. `scripts/qa.py` — QA

`scripts/qa.py` answers EgoLifeQA questions by retrieving from a build's `chroma_fio_opt`. Its `BUILD_CONFIG` must name the same stage ablation as the build it reads.

```bash
DB_MODE=opt BUILD_CONFIG=full MAX_DAY=7 EVIDENCE_K=5 python -u FIO/scripts/qa.py
```

Full documentation is in [`docs/qa_pipeline.md`](docs/qa_pipeline.md).

---

## 4. Supporting modules

| Module | Provides |
| --- | --- |
| `fio/llm.py` | Loads `Qwen/Qwen2.5-7B-Instruct` (bfloat16, `device_map="auto"`) at import; `call_qwen()`; the question-decomposition `system_message` |
| `fio/schema.py` | Field registry; `validate_fio`; `build_document_string`; `compute_heuristic_score` (field-presence score in [0, 1]); `compute_recency_for_day`; `parse_day_int` |
| `fio/storage/chroma.py` | `ChromaFIOStorage`: persistent client, `m_store_fio_entry`, `semantic_search` |
| `fio/storage/dual.py` | `DualFIOStorage` (raw + opt), `search` / `search_person_filter` / `search_speech_filter`, `get_where_filter` (DAY1…query day), `get_collection_name` (always `collection_day1_to_day7`) |
| `fio/storage/staging.py` | `StagedChromaDB`: `register()` + `sync_all()` for builds, `stage_in()` for read-only QA copies |
| `fio/tracking.py` | `BuildTracker`: per-phase timing, token counts, ETA, `build_report.json` (stdlib only) |
| `fio/stats.py` | `analyze_chroma_dates(path)`: entries per collection and per `date` |
| `fio/exp_logger.py` | `start_experiment(name)`: run logger |

Importing `Q_decompose` loads the 7B model onto the GPU. Any script that imports it — including both builds, even with Phase 1 skipped — needs a GPU, or enough RAM to hold the model on CPU.

---

## Check that everything resolves

Run this inside `FIO/`. It parses each file and checks its imports without running anything:

```bash
python - <<'EOF'
import ast, glob, importlib.util, sys
sys.path.insert(0, ".")
missing = set()
for f in glob.glob("fio/**/*.py", recursive=True) + glob.glob("scripts/*.py") + glob.glob("ablation/*.py"):
    for node in ast.walk(ast.parse(open(f).read(), f)):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for n in names:
            try:
                ok = importlib.util.find_spec(n) is not None
            except ModuleNotFoundError:
                ok = False
            if not ok:
                missing.add((f, n))
for f, n in sorted(missing):
    print(f"{f}: cannot find '{n}'")
print("done" if not missing else f"{len(missing)} unresolved")
EOF
```

## Known issues

1. The opt-store embedding patch may have no effect. `scripts/build_memory.py` swaps in its own document builder by reassigning `fio.schema.build_document_string`. But `fio/storage/dual.py` imported that function by name when it loaded, so `store_opt_fios()` still calls the original. Merged `action_sequence` text may therefore never reach the embeddings. To fix it, set `f["document"] = build_document_string_opt_aware(f)` on each kept FIO before `store_opt_fios()`. This changes the embeddings, so existing opt builds would need rebuilding.
2. The `scripts/build_memory.py` path defaults (`CAPTIONS_PATH`, `OUTPUT_DIR`, `FIO_CACHE_DIR`) point at A6_SHURE on the cluster; set them per run, or use `ablation/run.sh`, which derives them from `WEARER`.
3. Phase 1 samples (temperature 0.9). `SEED` makes a rerun repeatable on the same GPU and batch size, but a different GPU, driver or `BATCH_SIZE` can still change the extracted FIOs. Keep each run's `fio_cache/`: Phase 2 and QA are reproducible from it.