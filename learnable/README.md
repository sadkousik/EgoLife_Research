# AdaImprint experiment

This top-level package is an isolated research path for learnable FIO memory.
The existing `EgoRAG/FIO/fio_v3.py` build and QA pipeline remain unchanged and
remain the default.

## Current scaffold

- `ada_imprint.py` contains a recurrent controller, running feature
  normalization, and a self-supervised loss with both next-event prediction
  and sampled-past reconstruction.
- The controller emits a bounded log-importance correction and soft
  write/merge/update/discard preferences. A future memory adapter must still
  enforce the hard memory budget.
- FIO extraction and storage should be reused from `EgoRAG/FIO` through an
  explicit adapter rather than copied. Keep experiment configs, checkpoints,
  and generated memories under a separate `learnable/outputs/` location.

## Intended integration boundary

1. Read chronological FIO records produced by the existing extraction path.
2. Maintain prototype statistics separately from retained episodic entries.
3. Build controller features from prototype history, novelty, redundancy,
   prediction surprise, and budget occupancy.
4. Train offline on chronological chunks with truncated BPTT; validate against
   both next events and randomly sampled past events.
5. Keep inference frozen by default. Only an explicit online-learning option
   should update controller or predictor parameters during a build.

The current files are model components only; no existing pipeline invokes them
until an explicit experiment runner and adapter are added.
