# fio_v3.py
# ─────────────────────────────────────────────────────────────────────────────
# FIO Memory Build Pipeline — v3  (with batched GPU generation)
#
# Completely separate from fio.py — does NOT modify it.
#
# PARALLELISATION:
#   Your Qwen2.5-7B runs via HuggingFace transformers on your 48GB GPU.
#   model.generate() can process N captions simultaneously — the GPU
#   is mostly idle during single-caption generation (~15% utilisation).
#   Batching 4 captions at once → ~70% utilisation → 3-4x speedup.
#
#   TWO-PHASE BUILD:
#   Phase 1 — Batched LLM extraction
#     Process BATCH_SIZE captions per model.generate() call.
#     Save raw FIO JSON to disk per caption.
#     Embarrassingly parallel — captions are independent at this stage.
#     Time: ~2h (vs ~7.5h sequential)
#
#   Phase 2 — Sequential optimizer (NO hierarchy)
#     Load cached FIO JSON from disk. No LLM calls.
#     FIOOptimizer must stay sequential (stateful).
#     Time: ~30 min
#
#   Total: ~2.5h  vs  ~8h sequential
#
# HIERARCHY REMOVED:
#   The hierarchical-memory (slot/day digest) layer and everything used
#   only by it (HierarchicalMemory, slot/day narrative builders, narrative
#   prompts, HIER_PATH, ENABLE_NARRATIVES) have been stripped. This driver
#   now builds the flat FIO memory store only.
#
# STAGE ABLATION:
#   Set the ABLATION env var (or edit ABLATION below) to one of:
#     full   — all stages on (baseline anchor)
#     no_ctx — contextual encoding off (β=0)
#     no_s2  — importance pruning off (τ → keep-all)
#     no_s3  — interaction consolidation off (never merge)
#     no_s1  — prototype importance → uniform retention (NEEDS optimizer patch)
#   Each ablation writes to its own OUTPUT_DIR so runs never clobber.
#
# KEY SETTINGS:
#   BATCH_SIZE = 4   safe for 48GB (uses ~29GB)
#   BATCH_SIZE = 6   also fits (uses ~38GB), try after 4 is stable
#   BATCH_SIZE = 8   risky — long captions may OOM
#
# PATCHES APPLIED (opt FIO storage fixes):
#   OPT-1 — action_sequence content was not embedded in ChromaDB document.
#            Merged FIO event history was invisible to semantic search.
#            Fix: _action_sequence_to_text() + _enrich_opt_fios_for_storage()
#            appends compressed event text as opt_document_extra before storage.
#            build_document_string_opt_aware() includes it in the embedding.
#
#   OPT-2 — build_document_string replaced with opt-aware version that
#            includes opt_document_extra (action_sequence summary) in the
#            embedded document for merged FIOs. Backward-compatible: raw FIOs
#            produce identical output since opt_document_extra is absent.
# ─────────────────────────────────────────────────────────────────────────────

#---- file nneded---------------------------------------------------------------
#opti_fio_test.py  (FIOOptimizer — unchanged, unless running no_s1 ablation)
#fio_schema.py     (validate_fio, compute_heuristic_score, etc. — unchanged)

#-------------------------------------------------------------------------------



import json
import re
import time
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ── Existing files (unchanged) ────────────────────────────────────────────────
from Q_decompose import call_qwen, tokenizer, model   # local Qwen2.5-7B
from opti_fio_test import FIOOptimizer                # unchanged (except no_s1)

# ── New files ─────────────────────────────────────────────────────────────────
from prompt_fio_v2 import (
    FIO_EXTRACTION_PROMPT,
)
from fio_schema import (
    validate_fio,
    compute_heuristic_score,
    compute_recency_for_day,
    parse_day_int,
)
# NOTE: build_document_string is replaced by build_document_string_opt_aware
# below — do NOT import it from fio_schema here.
from build_tracker import BuildTracker
from chroma_fio_v2 import DualFIOStorage, get_collection_name

# ── Experiment logger (existing) ──────────────────────────────────────────────
from experiment_logger import start_experiment


# ══════════════════════════════════════════════════════════════════════════════
# Config — tune these for your GPU
# ══════════════════════════════════════════════════════════════════════════════

BATCH_SIZE: int = 6
# captions per model.generate() call
# 4 → ~29GB VRAM  (safe for 48GB)
# 6 → ~38GB VRAM  (also fits, try after 4 is stable)
# 8 → ~47GB VRAM  (risky on long captions)

MAX_INPUT_TOKENS: int = 3072
# truncate captions longer than this
# protects against KV cache OOM in batch mode


# ══════════════════════════════════════════════════════════════════════════════
# GPU setup — called once at startup
# ══════════════════════════════════════════════════════════════════════════════

def setup_tokenizer_for_batching():
    """
    Set padding_side = "left" on the tokenizer.
    REQUIRED for batched causal LM generation.

    Why left-padding:
      Causal LMs generate tokens left-to-right.
      The model must see real tokens at the END of each sequence.
      If you pad on the right (default), the model sees padding tokens
      before the real content and generates garbage for the padded items.
      Left-padding puts real tokens at the end — correct for generation.

    Call once before Phase 1 starts.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"[Setup] tokenizer.padding_side = 'left'  "
          f"pad_token_id = {tokenizer.pad_token_id}")


# ══════════════════════════════════════════════════════════════════════════════
# JSON extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_json_array(text: str) -> Optional[List[Dict]]:
    """Extract first JSON array from LLM output. Returns None on failure."""
    if not text:
        return None
    match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if not match:
        return None
    json_str = match.group(0)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    try:
        result = json.loads(json_str)
        return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Prompt builder — shared by both single and batch paths
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(caption: Dict) -> str:
    """Build the full formatted prompt string for one caption."""
    messages = [
        {"role": "system", "content": FIO_EXTRACTION_PROMPT},
        {"role": "user",   "content": caption["text"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# OPT PATCH: action_sequence → text for embedding  (OPT-1)
# ══════════════════════════════════════════════════════════════════════════════

def _action_sequence_to_text(action_sequence_str: str) -> str:
    """
    Convert an action_sequence JSON string into compact natural-language text
    suitable for inclusion in the ChromaDB embedded document.

    action_sequence format (produced by FIOOptimizer._merge_fios):
      '[{"action": "picked up", "object": "phone", "start_time": "...",
         "end_time": "...", "attributes": "{...}", "speech_content": "..."}, ...]'

    Returns e.g.:
      "also: picked up phone; looked at screen; said: \"time to go\""

    Empty / invalid sequences → "".
    """
    if not action_sequence_str or action_sequence_str in ("[]", "", "null"):
        return ""
    try:
        seq = json.loads(action_sequence_str)
        if not isinstance(seq, list) or len(seq) == 0:
            return ""
        parts = []
        for step in seq:
            if not isinstance(step, dict):
                continue
            act = (step.get("action") or "").strip()
            obj = (step.get("object") or "").strip()
            sp  = (step.get("speech_content") or "").strip()
            if act and obj:
                parts.append(f"{act} {obj}")
            elif act:
                parts.append(act)
            if sp:
                parts.append(f'said: "{sp[:60]}"')
        if not parts:
            return ""
        return "also: " + "; ".join(parts)
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""


def _enrich_opt_fios_for_storage(kept_fios: List[Dict]) -> None:
    """
    Mutates kept_fios IN PLACE before store_opt_fios() is called.

    For each FIO that has a non-empty action_sequence (i.e. a merged FIO),
    converts the compressed event history to natural-language text and stores
    it as opt_document_extra. build_document_string_opt_aware() then includes
    this in the ChromaDB embedded document so semantic search can match
    queries against merged event content — not just the single canonical field.

    For unmerged FIOs (action_sequence == "[]"), opt_document_extra = "".
    """
    for fio in kept_fios:
        seq_text = _action_sequence_to_text(fio.get("action_sequence", "[]"))
        fio["opt_document_extra"] = seq_text


# ══════════════════════════════════════════════════════════════════════════════
# OPT PATCH: opt-aware document string builder  (OPT-2)
# Replaces build_document_string from fio_schema for opt FIO storage.
# ══════════════════════════════════════════════════════════════════════════════

def build_document_string_opt_aware(fio: Dict) -> str:
    """
    Opt-aware replacement for fio_schema.build_document_string().

    For raw FIOs: identical output to the original (opt_document_extra absent).
    For opt FIOs: appends action_sequence summary so merged event history is
                  included in the semantic embedding.

    Field order mirrors the original build_document_string to keep embeddings
    comparable for ablation studies.
    """
    parts = []

    persons = (fio.get("persons") or "").strip()
    action  = (fio.get("action")  or "").strip()
    obj     = (fio.get("object")  or "").strip()
    tool    = (fio.get("tool")    or "").strip()
    app     = (fio.get("app")     or "").strip()
    src_loc = (fio.get("source_location") or "").strip()
    tgt_loc = (fio.get("target_location") or "").strip()
    speech  = (fio.get("speech_content")  or "").strip()

    if persons: parts.append(f"persons: {persons}")
    if action:  parts.append(f"action: {action}")
    if obj:     parts.append(f"object: {obj}")
    if tool:    parts.append(f"tool: {tool}")
    if app:     parts.append(f"app: {app}")
    if src_loc: parts.append(f"from: {src_loc}")
    if tgt_loc: parts.append(f"to: {tgt_loc}")
    if speech:  parts.append(f"speech: {speech}")

    # ── OPT-1: include merged action_sequence summary ─────────────────────────
    # opt_document_extra is set by _enrich_opt_fios_for_storage() at build time.
    # For raw FIOs this key is absent → empty string → nothing appended.
    extra = (fio.get("opt_document_extra") or "").strip()
    if not extra:
        # Fallback: compute on-the-fly if build-time enrichment was skipped.
        extra = _action_sequence_to_text(fio.get("action_sequence", "[]"))
    if extra:
        parts.append(extra)

    # ── Attributes (same as original) ─────────────────────────────────────────
    attrs_str = fio.get("attributes", "{}")
    try:
        attrs = json.loads(attrs_str) if isinstance(attrs_str, str) else attrs_str
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                if v:
                    parts.append(f"{k}: {v}")
    except (json.JSONDecodeError, TypeError):
        pass

    return " | ".join(parts) if parts else "no data"


# ══════════════════════════════════════════════════════════════════════════════
# FIO struct builder — shared by single and batch paths
# ══════════════════════════════════════════════════════════════════════════════

def _build_fios_from_json(
    fio_json: Optional[List[Dict]],
    caption: Dict,
) -> List[Dict]:
    """
    Convert raw LLM JSON output into validated FIO dicts for one caption.
    """
    if not fio_json:
        return []

    fios: List[Dict] = []
    for count, entry in enumerate(fio_json):
        if not isinstance(entry, dict):
            continue

        # Normalise persons
        persons_raw = entry.get("persons", "")
        if isinstance(persons_raw, list):
            persons_raw = ", ".join(str(p).strip() for p in persons_raw if p)
        persons = str(persons_raw).strip() if persons_raw else ""

        # Normalise attributes to JSON string
        attrs_raw = entry.get("attributes", {})
        if isinstance(attrs_raw, dict):
            attrs_str = json.dumps(attrs_raw, ensure_ascii=False)
        elif isinstance(attrs_raw, str):
            try:
                json.loads(attrs_raw)
                attrs_str = attrs_raw
            except (json.JSONDecodeError, ValueError):
                attrs_str = "{}"
        else:
            attrs_str = "{}"

        fio: Dict = {
            "id":               f"{caption['date']}-{caption['start_time']}-{count}",
            "date":             caption.get("date", ""),
            "start_time":       str(caption.get("start_time", "")),
            "end_time":         str(caption.get("end_time", "")),
            "video_path":       caption.get("video_path", ""),
            "persons":          persons,
            "action":           str(entry.get("action") or "").strip(),
            "object":           str(entry.get("object") or "").strip(),
            "tool":             str(entry.get("tool") or "").strip(),
            "app":              str(entry.get("app") or "").strip(),
            "source_location":  str(entry.get("source_location") or "").strip(),
            "target_location":  str(entry.get("target_location") or "").strip(),
            "speech_content":   str(entry.get("speech_content") or "").strip(),
            "attributes":       attrs_str,
            "heuristic_score":  0.0,
            "importance_score": 0.0,
            "recency_score":    0.0,
            "alpha_score":      0.5,
            "optimized":        False,
            "action_sequence":  "[]",
            # opt_document_extra is populated at Phase 2 store time, not here
        }
        fio["heuristic_score"]  = compute_heuristic_score(fio)
        fio["importance_score"] = fio["heuristic_score"]
        fios.append(fio)

    return fios


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Batched LLM extraction
# ══════════════════════════════════════════════════════════════════════════════

def generate_fio_batch(
    captions_batch: List[Dict],
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_new_tokens: int = 2200,
) -> Tuple[List[List[Dict]], Dict]:
    """
    Extract FIOs from a BATCH of captions in one model.generate() call.

    How batching works here:
      1. Build one formatted prompt string per caption
      2. tokenizer(list_of_prompts, padding=True) pads all prompts to the
         same length using LEFT-padding (set in setup_tokenizer_for_batching)
      3. One model.generate() processes all captions simultaneously on GPU
         GPU cores handle all batch items in parallel — 3-4x faster than
         calling model.generate() once per caption
      4. outputs shape: (batch_size, input_len + new_tokens)
         Slice outputs[i][input_len:] to get only new tokens for item i
      5. Decode and parse JSON for each item independently

    Returns:
        batch_fios: list of FIO lists, one per caption
        meta: timing + token stats for the whole batch
    """
    t0   = time.time()
    meta = {
        "input_tokens": 0, "output_tokens": 0,
        "latency_sec": 0.0, "tok_per_sec": 0.0,
        "n_success": 0, "n_fail": 0, "error": None,
    }

    prompts = [_build_prompt(c) for c in captions_batch]

    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]
        meta["input_tokens"] = input_len * len(captions_batch)

        if input_len > MAX_INPUT_TOKENS * 0.9:
            print(f"  [WARN] Batch input near token limit: {input_len}")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        batch_fios: List[List[Dict]] = []
        total_output_tokens = 0

        for i, caption in enumerate(captions_batch):
            new_token_ids = outputs[i][input_len:]
            response_text = tokenizer.decode(
                new_token_ids, skip_special_tokens=True
            )
            total_output_tokens += new_token_ids.shape[0]

            fio_json = extract_json_array(response_text)
            fios     = _build_fios_from_json(fio_json, caption)
            batch_fios.append(fios)

            if fios:
                meta["n_success"] += 1
            else:
                meta["n_fail"] += 1

        latency = time.time() - t0
        meta["output_tokens"] = total_output_tokens
        meta["latency_sec"]   = round(latency, 3)
        meta["tok_per_sec"]   = round(total_output_tokens / max(latency, 0.001), 1)

        return batch_fios, meta

    except torch.cuda.OutOfMemoryError:
        latency = time.time() - t0
        meta["latency_sec"] = round(latency, 3)
        meta["error"] = "OOM"
        print(f"  [OOM] batch_size={len(captions_batch)} → will retry single")
        torch.cuda.empty_cache()
        return [[] for _ in captions_batch], meta

    except Exception as e:
        latency = time.time() - t0
        meta["latency_sec"] = round(latency, 3)
        meta["error"] = str(e)
        print(f"  [ERROR] Batch generation failed: {e}")
        return [[] for _ in captions_batch], meta


def generate_fio_single(caption: Dict) -> Tuple[List[Dict], Dict]:
    """
    Single-caption fallback.
    Used when a batch OOMs or a caption is individually too long.
    """
    t0   = time.time()
    meta = {"input_tokens": 0, "output_tokens": 0, "latency_sec": 0.0, "error": None}

    prompt = _build_prompt(caption)
    try:
        inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        meta["input_tokens"] = input_len

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=2200,
                pad_token_id=model.config.pad_token_id,
            )

        new_token_ids = outputs[0][input_len:]
        response_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        meta["output_tokens"] = new_token_ids.shape[0]
        meta["latency_sec"]   = round(time.time() - t0, 3)

        fio_json = extract_json_array(response_text)
        return _build_fios_from_json(fio_json, caption), meta

    except Exception as e:
        meta["latency_sec"] = round(time.time() - t0, 3)
        meta["error"] = str(e)
        print(f"  [ERROR] Single generation failed: {e}")
        return [], meta


def run_phase1_extraction(
    captions:   List[Dict],
    cache_dir:  str,
    tracker:    BuildTracker,
    batch_size: int  = BATCH_SIZE,
    resume:     bool = True,
) -> int:
    """
    PHASE 1: Extract FIOs from ALL captions using batched GPU generation.
    Saves one JSON file per caption to cache_dir.

    resume=True lets you restart a crashed Phase 1 without re-doing
    captions that are already cached.
    """
    os.makedirs(cache_dir, exist_ok=True)

    n_total   = len(captions)
    n_done    = 0
    n_skipped = 0
    n_failed  = 0

    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Batched LLM extraction")
    print(f"  Captions : {n_total} | Batch size: {batch_size}")
    print(f"  Cache dir: {cache_dir}")
    if resume:
        print(f"  Resume   : ON — already-cached captions will be skipped")
    print(f"{'='*60}\n")

    i = 0
    while i < n_total:
        batch = captions[i : i + batch_size]

        if resume:
            pending = []
            for cap in batch:
                cap_id   = f"{cap['date']}-{cap['start_time']}"
                out_path = os.path.join(cache_dir, f"{cap_id}.json")
                if os.path.exists(out_path):
                    n_skipped += 1
                else:
                    pending.append(cap)
            if not pending:
                i += batch_size
                n_done += len(batch)
                continue
            actual_batch = pending
        else:
            actual_batch = batch

        tracker.start_gen()
        batch_fios, meta = generate_fio_batch(actual_batch, temperature=0.9)

        if meta.get("error") == "OOM":
            print(f"  [RETRY] Single-caption fallback for {len(actual_batch)} captions")
            batch_fios = []
            for cap in actual_batch:
                fios, single_meta = generate_fio_single(cap)
                batch_fios.append(fios)
                tracker.end_gen(
                    input_tokens  = single_meta["input_tokens"],
                    output_tokens = single_meta["output_tokens"],
                    success       = bool(fios),
                    error_type    = single_meta.get("error"),
                )
        else:
            tracker.end_gen(
                input_tokens  = meta["input_tokens"],
                output_tokens = meta["output_tokens"],
                success       = meta["n_success"] > 0,
                error_type    = meta.get("error"),
            )

        for cap, fios in zip(actual_batch, batch_fios):
            cap_id   = f"{cap['date']}-{cap['start_time']}"
            out_path = os.path.join(cache_dir, f"{cap_id}.json")
            with open(out_path, "w") as f:
                json.dump(fios, f, ensure_ascii=False)
            if fios:
                n_done += 1
            else:
                n_failed += 1

        processed = i + len(batch)
        print(
            f"[P1 {processed:5d}/{n_total}] "
            f"batch={len(actual_batch)} | "
            f"{meta['latency_sec']:.1f}s | "
            f"{meta['tok_per_sec']:.0f} tok/s | "
            f"ok={meta['n_success']} fail={meta['n_fail']} | "
            f"skip={n_skipped}"
        )

        i += batch_size

    print(f"\n[Phase 1 done] extracted={n_done} failed={n_failed} skipped={n_skipped}")
    return n_done


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Sequential optimizer (no hierarchy)
# ══════════════════════════════════════════════════════════════════════════════

def run_phase2_optimize(
    captions:          List[Dict],
    cache_dir:         str,
    day_number:        int,
    dual_storage:      DualFIOStorage,
    fio_optimizer:     FIOOptimizer,
    tracker:           BuildTracker,
) -> List[Dict]:
    """
    PHASE 2: Optimizer over Phase 1 cache. No LLM calls. No hierarchy.

    Why this must stay sequential:
      FIOOptimizer has internal running state that accumulates across captions:
        - prototypes list (EMA-updated each FIO)
        - history deque (sliding window of recent FIOs)
        - alpha_table (interaction strength learned across all clips)
        - object_interactions (per-object history)
      Each caption's FIOs depend on all previous captions' state.

    OPT PATCHES applied here:
      After _apply_recency_scores():
        1. _enrich_opt_fios_for_storage(all_kept) — adds opt_document_extra
           (action_sequence summary text) to every kept FIO so that
           build_document_string_opt_aware() includes it in the ChromaDB
           embedded document. Merged event history becomes searchable.

      DualFIOStorage.store_opt_fios now uses build_document_string_opt_aware
      instead of the original build_document_string — see the monkey-patch
      at the bottom of this function.
    """
    all_kept:       List[Dict]            = []
    all_raw_by_day: Dict[int, List[Dict]] = {}

    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Sequential optimizer (no hierarchy)")
    print(f"  Captions: {len(captions)} (no LLM calls)")
    print(f"{'='*60}\n")

    for count, caption in enumerate(captions):
        tracker.start_caption(count, day=caption.get("date", "DAY1"))

        # ── Load pre-generated FIOs from Phase 1 cache ───────────────────────
        cap_id   = f"{caption['date']}-{caption['start_time']}"
        fio_path = os.path.join(cache_dir, f"{cap_id}.json")

        tracker.start_gen()
        if os.path.exists(fio_path):
            try:
                with open(fio_path) as f:
                    raw_fios = json.load(f)
                tracker.end_gen(
                    input_tokens=0, output_tokens=0,
                    success=bool(raw_fios),
                    error_type=None if raw_fios else "empty_cache",
                )
            except (json.JSONDecodeError, OSError) as e:
                raw_fios = []
                tracker.end_gen(
                    input_tokens=0, output_tokens=0,
                    success=False, error_type="cache_read_error",
                )
                print(f"  [WARN] Cache read failed {cap_id}: {e}")
        else:
            raw_fios = []
            tracker.end_gen(
                input_tokens=0, output_tokens=0,
                success=False, error_type="cache_missing",
            )
            print(f"  [WARN] No cache for {cap_id}")

        # ── Store RAW FIOs (baseline DB) ──────────────────────────────────────
        tracker.start_opt()
        for f in raw_fios:
            f["optimized"] = False
        dual_storage.store_raw_fios(raw_fios, day_number)

        day_int = int(re.sub(r"[^0-9]", "", str(caption.get("date", "1"))) or "1")
        all_raw_by_day.setdefault(day_int, []).extend(raw_fios)

        # ── FIO optimizer ─────────────────────────────────────────────────────
        prev_kept       = fio_optimizer._stats["total_kept"]
        prev_dropped    = fio_optimizer._stats["total_dropped"]
        prev_compressed = fio_optimizer._stats["total_compressed"]

        opt_result  = fio_optimizer.process(raw_fios)
        kept_output = opt_result["kept"]
        new_kept    = kept_output[prev_kept:]
        alpha_table = opt_result.get("alpha_table", {})

        delta_dropped    = fio_optimizer._stats["total_dropped"]    - prev_dropped
        delta_compressed = fio_optimizer._stats["total_compressed"] - prev_compressed

        for fio in new_kept:
            action = (fio.get("action") or "").lower().strip()
            fio["alpha_score"]   = alpha_table.get(action, {}).get("alpha", 0.5)
            fio["optimized"]     = True
            fio["recency_score"] = 0.0   # filled after loop

        tracker.end_opt(
            n_raw        = len(raw_fios),
            n_kept       = len(new_kept),
            n_compressed = delta_compressed,
            n_dropped    = delta_dropped,
        )

        # ── Hierarchy update REMOVED ──────────────────────────────────────────
        # The slot/day digest layer was stripped. The tracker hier-phase
        # bracket is kept (zero work) so per-caption phase accounting in
        # BuildTracker stays consistent with its expected start/end calls.
        tracker.start_hier()
        tracker.end_hier()

        all_kept.extend(new_kept)
        tracker.end_caption()

    # ── Recency scores ────────────────────────────────────────────────────────
    _apply_recency_scores(all_kept, all_raw_by_day)

    # ── OPT-1: Enrich opt FIOs with action_sequence text for embedding ────────
    # Must run AFTER _apply_recency_scores() and BEFORE store_opt_fios().
    # Adds opt_document_extra to every kept FIO so that merged event history
    # (action_sequence) is included in the ChromaDB embedded document.
    _enrich_opt_fios_for_storage(all_kept)
    n_merged = sum(1 for f in all_kept if f.get("opt_document_extra", ""))
    print(f"[OPT-1] Enriched {n_merged}/{len(all_kept)} merged FIOs "
          f"with action_sequence text for embedding")

    # ── OPT-2: Monkey-patch DualFIOStorage to use opt-aware document builder ──
    # DualFIOStorage.store_opt_fios() internally calls build_document_string().
    # We swap it to build_document_string_opt_aware so the embedded document
    # includes opt_document_extra (the merged event history).
    # This is a targeted patch — store_raw_fios is NOT affected.
    import chroma_fio_v2 as _cfv2
    import fio_schema as _fs
    _fs_original_doc_fn = _fs.build_document_string        # save original
    _fs.build_document_string = build_document_string_opt_aware   # swap in opt-aware

    # ── Store optimised FIOs ──────────────────────────────────────────────────
    dual_storage.store_opt_fios(all_kept, day_number)

    # Restore original so raw storage paths are unaffected
    _fs.build_document_string = _fs_original_doc_fn

    return all_kept


# ══════════════════════════════════════════════════════════════════════════════
# Recency scores
# ══════════════════════════════════════════════════════════════════════════════

def _apply_recency_scores(
    kept_fios:   List[Dict],
    raw_by_day:  Dict[int, List[Dict]],
):
    recency_map: Dict[str, float] = {}
    for day_int, day_raw_fios in raw_by_day.items():
        recency_map.update(compute_recency_for_day(day_raw_fios, day_int))
    for fio in kept_fios:
        fid = fio.get("id", "")
        if fid in recency_map:
            fio["recency_score"] = recency_map[fid]
        imp = fio.get("importance_score", 0.0)
        rec = fio.get("recency_score", 0.0)
        if rec > 0.0:
            fio["importance_score"] = round(imp * rec, 6)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import inspect

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE ABLATION SELECTOR
    # ══════════════════════════════════════════════════════════════════════════
    # Choose via env (ABLATION=no_s3 python fio_v3.py) or edit the default.
    # Each entry overrides FIOOptimizer kwargs on top of BASE_OPT_KWARGS and
    # gets its own OUTPUT_DIR so runs never overwrite one another.
    #
    # IMPORTANT: every row must hold the SAME config that produced the locked
    # full-model number, changing only the one factor under study. Only edit
    # BASE_OPT_KWARGS if you also re-run "full".
    ABLATION = os.environ.get("ABLATION", "full")

    # ── HYPERPARAMETER SWEEP (orthogonal to ABLATION) ──────────────────────────
    # Vary ONE BASE_OPT_KWARGS knob without touching the ablation config. Used for
    # the δ_g / prototype-cap / history-size sensitivity sweeps. Applied AFTER the
    # ablation override and reflected in OUTPUT_DIR so values never clobber. e.g.:
    #   SWEEP_PARAM=delta_g_threshold SWEEP_VAL=0.65  python fio_v3.py
    #   SWEEP_PARAM=max_prototypes    SWEEP_VAL=300   python fio_v3.py
    #   SWEEP_PARAM=history_window    SWEEP_VAL=10    python fio_v3.py
    # Value is cast to the type of the existing default (int stays int, float float).
    SWEEP_PARAM = os.environ.get("SWEEP_PARAM", "")
    SWEEP_VAL   = os.environ.get("SWEEP_VAL", "")

    # Unique tag for THIS run = ablation name (+ sweep value if sweeping).
    # Used for OUTPUT_DIR *and* the t-SNE / merge-stats export filenames, so a
    # parallel sweep can never collide or race on a shared file.
    # RUN_TAG = ABLATION + (f"__{SWEEP_PARAM}={SWEEP_VAL}" if SWEEP_PARAM else "")
    # line ~815  — make RUN_TAG overridable (keeps t-SNE/merge exports per-window)
    RUN_TAG = os.environ.get("RUN_TAG",
                ABLATION + (f"__{SWEEP_PARAM}={SWEEP_VAL}" if SWEEP_PARAM else ""))

    ABLATIONS = {
        "full": {
            "desc": "All stages on (baseline anchor).",
            "opt":  {},
        },
        "no_ctx": {
            # Stage 1 sub-component: contextual encoding. β=0 → ẽ_t = e_t,
            # so the history buffer / δ_h no longer influence clustering.
            "desc": "Contextual encoding off (β = 0).",
            "opt":  {"context_history_weight": 0.0},
        },
        "no_s2": {
            # Stage 2: importance pruning. w(f_t) > 0 always, so τ = -1.0
            # keeps every FIO. w(f_t) is still COMPUTED, so w_imp ranking in
            # retrieval still has values (this is the −S2a / pruning-only row).
            "desc": "Importance pruning off (τ → keep-all; w_imp still computed).",
            "opt":  {"tau_threshold": -1.0},
        },
        "no_s3": {
            # Stage 3: consolidation. The ctx-sim gate needs sim >= COMPRESSION_SIM;
            # cosine sim maxes at 1.0, so 2.0 means no FIO is ever merged — every
            # retained FIO becomes its own memory entry.
            "desc": "Interaction consolidation off (never merge).",
            "opt":  {"compression_sim_threshold": 2.0},
        },
        "no_s1": {
            # Stage 1: prototype-based importance. Clean ablation = replace the
            # structured w(f_t) keep/drop with UNIFORM retention at the SAME
            # keep-rate as the full model. This CANNOT be done from the driver —
            # it needs a small flag in opti_fio_test.py (see note below).
            # target_keep_rate must be set to the full model's measured
            # total_kept / total_input.
            "desc": "Prototype importance → uniform retention at matched keep-rate "
                    "(REQUIRES opti_fio_test.py patch).",
            "opt":  {"importance_mode": "uniform", "target_keep_rate": 0.4},
        },
        "s3_only": {
            # Only S3 (consolidation) active. S2 off → keep everything
            # (uniform draw at p=1.0 always keeps). S1 off → stored importance
            # neutralized, so no prototype signal reaches retrieval. S3 on →
            # default compression_sim, so redundant FIOs still merge. Isolates
            # what consolidation alone does to a fully-unfiltered store.
            # NOTE: target_keep_rate is fixed at 1.0 by design (keep-all) — the
            # full-model keep-rate caveat from no_s1 does NOT apply here.
            "desc": "Only consolidation active (S1 off, S2 off, keep-all + merge).",
            "opt":  {"importance_mode": "uniform", "target_keep_rate": 1.0,
                     "compression_sim_threshold": 0.75},
        },
    }

    if ABLATION not in ABLATIONS:
        sys.exit(f"[ABLATION] Unknown '{ABLATION}'. Choose: {list(ABLATIONS)}")

    abl = ABLATIONS[ABLATION]
    print(f"\n{'#'*60}\n# ABLATION = {ABLATION}: {abl['desc']}\n{'#'*60}")

    job_name = os.environ.get("SLURM_JOB_NAME", f"Build_7day_{ABLATION}")
    job_id   = os.environ.get("SLURM_JOB_ID", "")
    exp_name = f"{job_name}_{job_id}" if job_id else job_name
    logger   = start_experiment(exp_name)

    # ── Config ────────────────────────────────────────────────────────────────
    #CAPTIONS_PATH     = "/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoGPT/meta_7days_TASHA_13042026_1.json" #tasha
    CAPTIONS_PATH     = "/home/rs/24CS92R01/EGOLIFE/EgoLife_Research/generated_caption/A6_SHURE_meta.json"
    # OUTPUT_DIR        = f"./BUILD_7day_tau_{RUN_TAG}"   # unique per run
    OUTPUT_DIR    = os.environ.get("OUTPUT_DIR", f"/home/rs/24CS92R01/EGOLIFE/EgoLife_Research/FIO_mem/A6_SHURE_{RUN_TAG}")
    # FIO_CACHE_DIR     = os.path.join("memory_build_v2_7days", "fio_cache")
    FIO_CACHE_DIR = os.environ.get("FIO_CACHE_DIR",
                  os.path.join("/home/rs/24CS92R01/EGOLIFE/EgoLife_Research/FIO_mem/A6_SHURE", "fio_cache"))
    RAW_DB_DIR        = os.path.join(OUTPUT_DIR, "chroma_fio_raw")
    OPT_DB_DIR        = os.path.join(OUTPUT_DIR, "chroma_fio_opt")
    REPORT_PATH       = os.path.join(OUTPUT_DIR, "build_report.json")

    # TARGET_DAY        = 7
    TARGET_DAY    = int(os.environ.get("TARGET_DAY", "7"))
    TEST_DAY1_ONLY    = False
    SKIP_PHASE1       = False 

    # Build the raw (unoptimized) FIO store too?  The raw store is identical
    # across all ablations, so build it ONCE (e.g. on your full/baseline run)
    # and skip it for ablation sweeps to save time + disk. store_opt is always on.
    # Flip the default below, or override per-run:  STORE_RAW=1 python fio_v3.py
    STORE_RAW         = os.environ.get("STORE_RAW", "False").lower() in ("1", "true", "yes")
    print(f"[config] STORE_RAW = {STORE_RAW}  (raw FIO store {'ON' if STORE_RAW else 'SKIPPED'})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    #added to fix corrupt
    # ── Stage Chroma DBs on node-local disk (NFS-corruption fix) ──────────────
    from nfs_db_staging import StagedChromaDB
    _stager     = StagedChromaDB(local_root=os.environ.get("LOCAL_DB_ROOT"))
    RAW_DB_LIVE = _stager.register("chroma_fio_raw", RAW_DB_DIR if STORE_RAW else None)
    OPT_DB_LIVE = _stager.register("chroma_fio_opt", OPT_DB_DIR)
    #added end

    # ── Load captions ─────────────────────────────────────────────────────────
    with open(CAPTIONS_PATH) as f:
        all_captions = json.load(f)

    if TEST_DAY1_ONLY:
        captions   = [c for c in all_captions if c.get("date") == "DAY1"]
        TARGET_DAY = 1
        print(f"[TEST] DAY1 only: {len(captions)} captions")
    else:
        valid_days = {f"DAY{i}" for i in range(1, TARGET_DAY + 1)}
        captions   = [c for c in all_captions if c.get("date") in valid_days]
        print(f"Processing DAY1–DAY{TARGET_DAY}: {len(captions)} captions")

    # ── Initialise ────────────────────────────────────────────────────────────
    setup_tokenizer_for_batching()

    tracker = BuildTracker(
        model_name="Qwen2.5-7B-Instruct-local",
        verbose_progress=True,
        eta_window=20,
    )
    dual_storage = DualFIOStorage(
        raw_db_dir=RAW_DB_LIVE or RAW_DB_DIR,   # local during the run
        opt_db_dir=OPT_DB_LIVE,                 # local during the run
        store_raw=STORE_RAW, store_opt=True,
    )

    # ── FIOOptimizer kwargs: base config + ablation override ──────────────────
    # BASE_OPT_KWARGS is the locked full-model configuration. The ablation
    # dict overrides exactly one knob (or none, for "full").
    BASE_OPT_KWARGS = dict(
        delta_g_threshold          = 0.75,   # δ_g prototype-assignment similarity
        tau_threshold              = 0.2,    # τ importance keep threshold 0.2 default
        delta_h_threshold          = 0.3,    # δ_h history similarity filter (Stage-1 ctx) 0.3 default
        lambda_decay               = 1.146e-6,#0.01,
        redundancy_cooldown_sec    = 1.0,
        context_history_weight     = 0.5,    # β history weight in ẽ_t #at 0.5 got best result
        compression_sim_threshold  = 0.75,   # Stage-3 consolidation ctx-sim gate 0.75 default
        alpha_cold_start           = 0.5,    # unused (α removed) — harmless
        alpha_ema                  = 0.2,    # unused — harmless
        history_window             = 5,#10,
        max_prototypes             = 300, #200,    # N_max chnaged to 1000 for delta_g test
        max_action_prototypes      = 50,     # unused — harmless
        max_object_interactions    = 20,
        max_kept_for_compression   = 100,
        max_alpha_clusters         = 50,     # unused — harmless
        track_growth               = True,
    )
    opt_kwargs = {**BASE_OPT_KWARGS, **abl["opt"]}

    # ── Apply hyperparameter sweep override (if any) ──────────────────────────
    if SWEEP_PARAM:
        if SWEEP_PARAM not in BASE_OPT_KWARGS:
            sys.exit(f"[SWEEP] '{SWEEP_PARAM}' is not a BASE_OPT_KWARGS knob. "
                     f"Choose one of: {sorted(BASE_OPT_KWARGS)}")
        if SWEEP_VAL == "":
            sys.exit(f"[SWEEP] SWEEP_PARAM={SWEEP_PARAM} set but SWEEP_VAL is empty.")
        _t = type(BASE_OPT_KWARGS[SWEEP_PARAM])              # cast to the default's type
        try:
            _casted = (_t(SWEEP_VAL) if _t is not bool
                       else SWEEP_VAL.lower() in ("1", "true", "yes"))
        except ValueError:
            sys.exit(f"[SWEEP] could not cast SWEEP_VAL='{SWEEP_VAL}' to {_t.__name__} "
                     f"for {SWEEP_PARAM}.")
        opt_kwargs[SWEEP_PARAM] = _casted
        print(f"[SWEEP] {SWEEP_PARAM} = {_casted!r}  (overrides base {BASE_OPT_KWARGS[SWEEP_PARAM]!r})")

    # Guard: if an ablation passes kwargs FIOOptimizer doesn't accept (no_s1),
    # fail loudly instead of crashing with a cryptic TypeError downstream.
    _accepted = set(inspect.signature(FIOOptimizer.__init__).parameters)
    _unknown  = [k for k in opt_kwargs if k not in _accepted]
    if _unknown:
        sys.exit(
            f"[ABLATION:{ABLATION}] FIOOptimizer does not accept {_unknown}.\n"
            f"  This row needs a companion patch in opti_fio_test.py "
            f"(e.g. an `importance_mode='uniform'` branch in process() that "
            f"keeps each retained-eligible FIO with probability target_keep_rate "
            f"instead of using w(f_t) >= τ). Set target_keep_rate to the full "
            f"model's measured total_kept/total_input."
        )

    fio_optimizer = FIOOptimizer(**opt_kwargs)

    tracker.start_build(total_captions=len(captions))

    # ── PHASE 1: Batched LLM extraction ──────────────────────────────────────
    if not SKIP_PHASE1:
        run_phase1_extraction(
            captions   = captions,
            cache_dir  = FIO_CACHE_DIR,
            tracker    = tracker,
            batch_size = BATCH_SIZE,
            resume     = True,
        )
    else:
        print(f"[SKIP_PHASE1] Using existing cache: {FIO_CACHE_DIR}")

    # ── PHASE 2: Sequential optimizer (no hierarchy) ──────────────────────────
    all_kept = run_phase2_optimize(
        captions          = captions,
        cache_dir         = FIO_CACHE_DIR,
        day_number        = TARGET_DAY,
        dual_storage      = dual_storage,
        fio_optimizer     = fio_optimizer,
        tracker           = tracker,
    )
    # ── Optional viz exports (non-fatal) ──────────────────────────────────────
    # export_tsne_data / export_merge_statistics internally prepend "./t_sne/",
    # so pass a BARE filename (no subdirectory) — the ablation suffix keeps the
    # files distinct inside the shared ./t_sne/ dir. Wrapped in try/except because
    # the build + Chroma store are already complete by this point; a viz-export
    # hiccup must not fail an otherwise-successful build.
    os.makedirs("t_sne", exist_ok=True)
    try:
        fio_optimizer.export_tsne_data(f"prototype_tsne_data_{RUN_TAG}.json")
        fio_optimizer.export_merge_statistics(f"merge_stats_{RUN_TAG}.json")
    except Exception as e:
        print(f"[WARN] viz export failed (non-fatal) for ablation={ABLATION}: {e}")

    # ── Save ──────────────────────────────────────────────────────────────────
    # hier_stats removed (no hierarchy). optimizer_stats carries the ablation
    # diagnostics you care about: total_kept / total_dropped / total_compressed
    # / proto_cap_hits.
    tracker.finish_build(hier_stats={}, optimizer_stats=fio_optimizer._stats)
    tracker.save_report(REPORT_PATH)
    dual_storage.show_stats()
    #added to fix corrupt
    # Flush node-local Chroma DBs back to NFS (whole-file copy, corruption-safe)
    _stager.sync_all()
    #end

    print(f"\n[fio_v3] Complete — ABLATION={ABLATION}")
    print(f"  Kept FIOs : {len(all_kept)}")
    print(f"  Opt stats : {fio_optimizer._stats}")
    print(f"  FIO cache : {FIO_CACHE_DIR}/")
    print(f"  Output dir: {OUTPUT_DIR}/")
    print(f"  Report    : {REPORT_PATH}")

    #print entries in the FIO
    from fio_count import analyze_chroma_dates
    if STORE_RAW:
        print(f"\nIn Raw DB:{RAW_DB_DIR}")
        analyze_chroma_dates(RAW_DB_DIR)
    else:
        print(f"\n[STORE_RAW=False] raw DB skipped — not analyzed")

    print(f"\nIn Opt DB{OPT_DB_DIR}")
    analyze_chroma_dates(OPT_DB_DIR)

    logger.info(f"fio_v3 build complete — ablation={ABLATION}")