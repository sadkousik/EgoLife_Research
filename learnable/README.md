# AdaImprint — learnable FIO memory

AdaImprint replaces the hand-set retention rule of the FIO memory
(`FIO/fio/optimizer.py`) with learned parameters: the importance formula, a
recurrent controller, a write/merge/update policy and the consolidation
threshold, trained self-supervised on the interaction stream.

`FIO/` is not modified. AdaImprint subclasses `FIOOptimizer`, imports the FIO
storage modules, and provides its own Phase-2 build. The model is described
in [docs/method.md](docs/method.md).

```
learnable/
├── adaimprint/               # importable library (no entry points)
│   ├── controller.py         # LSTM controller (δ_t, policy logits), running feature norm
│   ├── model.py              # learnable importance, consolidation gate, policy, predictor; checkpoints
│   ├── memory.py             # slot memory, read-out, StreamProcessor (steps 3–8 on one stream), losses
│   ├── data.py               # Phase-1 cache iterator, trace loading, day split, QA evidence
│   ├── training.py           # run a trace (train / eval), comparison with the heuristic
│   └── fio_adapter.py        # LearnableFIOOptimizer(FIOOptimizer): build-time decisions, eviction, traces
├── scripts/                  # entry points (python -m learnable.scripts.<name>)
│   ├── record_trace.py       # replay a Phase-1 cache through the heuristic -> trace (CPU, no LLM)
│   ├── train.py              # train + validate, writes best.pt / last.pt / history.json
│   └── build_memory.py       # Phase-2 build with a checkpoint (or the heuristic) -> Chroma stores
├── jobs/run_learnable.sh     # trace -> train -> build pipeline
├── docs/method.md            # the model in mathematical form
├── tests/test_smoke.py       # offline end-to-end test (stub encoder in tests/stub/)
└── outputs/                  # traces and checkpoints (git-ignored)
```

## Run

Run everything from the repository root:

```bash
python learnable/tests/test_smoke.py                    # ~1 min, offline

bash learnable/jobs/run_learnable.sh trace              # FIO_MEM_ROOT=<root>/<wearer>/fio_cache
bash learnable/jobs/run_learnable.sh train              # MODE=params: only the scalar parameters
bash learnable/jobs/run_learnable.sh build              # BUILD_ARGS="--budget N --online-lr x --doc-fix"

# one wearer: train on DAY1-5, validate on DAY6-7
WEARERS="A1_JAKE" SPLIT_DAY=6 EPOCHS=3 bash learnable/jobs/run_learnable.sh all
# several wearers: leave VAL_WEARER out
WEARERS="A4_LUCIA A5_KATRINA" VAL_WEARER=A5_KATRINA bash learnable/jobs/run_learnable.sh all
```

The steps can also be run on their own:

```bash
python -m learnable.scripts.record_trace --wearer A1_JAKE \
    --captions generated_caption/A1_JAKE_meta.json \
    --cache-dir FIO_mem/A1_JAKE/fio_cache --out learnable/outputs/traces/A1_JAKE.pt
python -m learnable.scripts.train --train learnable/outputs/traces/A1_JAKE.pt \
    --split-day 6 --out learnable/outputs/ckpt_full_A1_JAKE_split6
python -m learnable.scripts.build_memory --ckpt learnable/outputs/ckpt_full_A1_JAKE_split6/best.pt \
    --captions generated_caption/A1_JAKE_meta.json \
    --cache-dir FIO_mem/A1_JAKE/fio_cache --out FIO_mem/A1_JAKE_learnable_full
```

Traces store everything training needs, so training never touches the
encoder or the LLM. Re-record a trace whenever a FIO clustering knob changes
(δ_g, δ_h, β, `max_prototypes`) or `train.py` reports that it was recorded with
another code version.

## What is learned

| Hand-set in FIO | Learned in AdaImprint | Where |
| --- | --- | --- |
| I_t = log(1+ν) · e^(−λΔt) / (1+log(1+N)) | exponents a_F, a_D, decay λ, bias b | `model.LearnableImportance.base_log_w` |
| — | context correction δ_t ∈ [−2, 2] from an LSTM | `controller.AdaImprintController` |
| keep iff I_t ≥ τ | soft gate σ((log Î_t − log τ)/T) | `model.LearnableImportance.step` |
| always merge into the equivalent entry | policy over write / merge / update | `model.policy_probs` |
| equivalent iff cos ≥ 0.75 (`compression_sim_threshold`) | threshold θ_c ∈ [0.5, 1] | `model.LearnableImportance.cons_thr` |
| unbounded memory | hard budget B with evict-or-consolidate | `fio_adapter.LearnableFIOOptimizer` |

At initialisation every learned quantity equals its heuristic value, and the
model makes exactly the heuristic's decisions. The smoke test checks this, and
epoch 0 of every training run reports it as the baseline.

`--mode params` learns only the scalars (a_F, a_D, λ, b, θ_c). `--fix-cons`
keeps θ_c at 0.75.

## Algorithm → code

| Step | Where |
| --- | --- |
| 1 Encode f_t, embedding e_t | `FIOOptimizer.encode_with_context` (unchanged) |
| 2 Match / create prototype | `FIOOptimizer.assign_prototype`; the adapter also records the pre-assignment similarity |
| 3 F_t, Δ_t, D_t, novelty, history features | `fio_adapter._features` (from the prototype bank **before** its update) |
| 3 Redundancy U_t, occupancy b_t | `memory.StreamProcessor.step` |
| 4 Predictive surprise E_t | `model.surprise_from_logits`; the predictor reads the compressed memory |
| 5 LSTM, correction δ_t, policy | `controller.AdaImprintController`, `model.LearnableImportance.step` |
| 6 Consolidation candidate | `fio_adapter._find_candidate`: best same-pattern entry down to the floor; θ_c decides |
| 6 Hard budget B | `fio_adapter._lowest_utility`, `_evict` (utility = Î · e^(−λ·age)) |
| 7 Prototype statistics for every FIO | unchanged parent behaviour; kept separate from the memory |
| 8 Losses, truncated BPTT | `memory.StreamProcessor.chunk_loss`, `training.run_trace` |
| 8 Online learning | `scripts/build_memory.py --online-lr x` (frozen by default) |

## Training objective

```
L = L_pred + w_past·L_past + β·L_mem + γ·L_redund   [+ w_ev·L_evidence]
```

* `L_pred`: next-prototype cross-entropy from the controller state and a memory read-out.
* `L_past`: recover the prototype of random past FIOs from the memory, read by
  content and time. This is the guard against keeping only routine,
  predictable events. A time-only read (`--past-read time`) does not work:
  the FIOs of one caption clip share a timestamp, and on A1_JAKE it scored at
  chance whatever was kept.
* `L_mem`: |write rate − r*| plus a small merge/update cost; r* defaults to the
  heuristic's write rate, so memories are compared at equal size.
* `L_redund`: penalises writing near-duplicate neighbours.
* `L_evidence` (optional): EgoLifeQA `target_time` of training wearers only.

The memory is built from the hard decisions; gradients reach the gate, the
policy and θ_c through a straight-through estimator.

Each validation line reports the learned model's discard / write / merge /
update rates, `next_acc` and `recon_acc` / `recon_ce`, then `heur_*` (the
heuristic at τ) and `heur_matched_*` (the heuristic at the learned model's
memory size).

## Building for QA

`scripts/build_memory.py` replaces Phase 2 of `FIO/scripts/build_memory.py`. It
reads the Phase-1 cache and needs no LLM or GPU; Phase-1 extraction still runs
through FIO. Without `--ckpt` it builds the heuristic exactly as FIO's `full`
row does.

| Option | Effect |
| --- | --- |
| `--budget N` | hard memory budget (eviction) |
| `--online-lr x` | keep training during the build; `--save-ckpt` keeps the result |
| `--doc-fix` | embed merged `action_sequence` text in the opt store (see below) |
| `--opt key=value` | override an optimizer knob, as `SWEEP_PARAM` does in FIO |
| `--store-raw`, `--local-db-root` | same as `STORE_RAW` / `LOCAL_DB_ROOT` |

The build stores `final_memory()`, the live memory after the stream, rather
than the per-caption copies FIO collects (later merges into earlier entries are
lost there). Evaluate with `QA_BUILD_DIR=<out> python -u FIO/scripts/qa.py`.

`--doc-fix`: in FIO builds the merged history never reaches the embeddings,
because both storage modules bind `build_document_string` by name and the
Chroma writer rebuilds the document. `--doc-fix` patches both while the opt
store is written. It is off by default so learned and heuristic builds embed
documents the same way.

## Limitations

* `L_past` predicts prototype ids, which only approximate "what happened then".
  Confirm gains with QA on the held-out wearer.
* Candidates are recorded under the heuristic's memory. During training a
  candidate counts only if the learned memory kept it; entries the heuristic
  dropped are never offered as merge targets.
* The pattern test for consolidation (same action|object|app, synonym groups,
  the 0.60 verb-similarity fallback) stays fixed; only the similarity
  threshold is learned.
* Δt comes from FIO's `_effective_ts`, which adds a running FIO counter to the
  clip time (about 19 h by the end of a wearer's stream), so the learned λ is
  fitted to that clock.
* Eviction scans the memory linearly, which is fine for B in the thousands.
