# fio_v3_abl.py — ABLATION VERSION of fio_v3.py
# Drop-in replacement that reads λ and τ from environment variables
# (ABLATION_LAMBDA, ABLATION_TAU, ABLATION_TAG) so run_ablation_abl.py
# can sweep them. Defaults preserve legacy behavior when env vars are unset.
# ─────────────────────────────────────────────────────────────────────────────
# FIO Memory Build Pipeline — v2  (with batched GPU generation)
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
#   Phase 2 — Sequential optimizer + hierarchy
#     Load cached FIO JSON from disk. No LLM calls.
#     FIOOptimizer + HierarchicalMemory must stay sequential (stateful).
#     Time: ~30 min
#
#   Total: ~2.5h  vs  ~8h sequential
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
from opti_fio_abl import FIOOptimizer                # ablation-capable version
from fio_hierarchy import HierarchicalMemory          # unchanged

# ── New files ─────────────────────────────────────────────────────────────────
from prompt_fio_v2 import (
    FIO_EXTRACTION_PROMPT,
    SLOT_NARRATIVE_PROMPT,
    DAY_NARRATIVE_PROMPT,
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from EgoRAG.experiment_logger import start_experiment


# ══════════════════════════════════════════════════════════════════════════════
# Ablation hooks — env-driven so run_ablation.py can sweep λ and τ
# ══════════════════════════════════════════════════════════════════════════════
ABLATION_TAG    = os.environ.get("ABLATION_TAG", "Jake_7daysFinal")
ABLATION_LAMBDA = float(os.environ.get("ABLATION_LAMBDA", "8.0e-6"))  # 1-day half-life
ABLATION_TAU    = float(os.environ.get("ABLATION_TAU",    "0.2"))
#added by me
CACHE_DIR= os.environ.get("CACHE_DIR", "./memory_build_v2_7days") #memory_build_v2_7days", "fio_cache"

# Component ablation flags (Study B). Set to "1" to remove that term from w(f_i).
DISABLE_FREQUENCY       = os.environ.get("DISABLE_FREQUENCY",       "0") == "1"
DISABLE_RECENCY         = os.environ.get("DISABLE_RECENCY",         "0") == "1"
DISABLE_DISTINCTIVENESS = os.environ.get("DISABLE_DISTINCTIVENESS", "0") == "1"
DISABLE_INTERACTION     = os.environ.get("DISABLE_INTERACTION",     "0") == "1"


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
# PHASE 2 — Sequential optimizer + hierarchy
# ══════════════════════════════════════════════════════════════════════════════

def run_phase2_optimize(
    captions:          List[Dict],
    cache_dir:         str,
    day_number:        int,
    dual_storage:      DualFIOStorage,
    fio_optimizer:     FIOOptimizer,
    hier_memory:       HierarchicalMemory,
    tracker:           BuildTracker,
    enable_narratives: bool = False,
) -> List[Dict]:
    """
    PHASE 2: Optimizer + hierarchy over Phase 1 cache. No LLM calls.

    Why this must stay sequential:
      FIOOptimizer has internal running state that accumulates across captions:
        - prototypes list (EMA-updated each FIO)
        - history deque (sliding window of recent FIOs)
        - alpha_table (interaction strength learned across all clips)
        - object_interactions (per-object history)
      Each caption's FIOs depend on all previous captions' state.

      HierarchicalMemory tracks a clip counter per day to detect
      slot (1-hour) and day boundaries — must see clips in order.

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

    # ── Optional LLM narrative hook ───────────────────────────────────────────
    def llm_hook(digest: Dict, level: str) -> str:
        if not enable_narratives:
            return ""
        if level == "slot":
            return _build_slot_narrative(digest)
        elif level == "daily":
            day_int = digest.get("day", 0)
            slot_narratives = [
                s.get("llm_summary", "")
                for (d, _), s in hier_memory._slots.items()
                if d == day_int and s.get("llm_summary")
            ]
            return _build_day_narrative(digest, slot_narratives)
        return ""

    if enable_narratives:
        hier_memory._llm_fn = llm_hook

    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Sequential optimizer + hierarchy")
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

        # ── Hierarchy update ──────────────────────────────────────────────────
        tracker.start_hier()
        hier_memory.update(
            new_kept,
            clip_date=caption["date"],
            alpha_table=alpha_table,
        )
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

    # ── Finalise hierarchy ────────────────────────────────────────────────────
    hier_memory.finalise()

    return all_kept


# ══════════════════════════════════════════════════════════════════════════════
# Narrative helpers (used when enable_narratives=True)
# ══════════════════════════════════════════════════════════════════════════════

def _build_slot_narrative(slot_digest: Dict) -> str:
    high_alpha = slot_digest.get("high_alpha_events", [])[:5]
    summary_input = high_alpha if high_alpha else {
        "top_actions": slot_digest.get("top_actions", []),
        "top_objects": slot_digest.get("top_objects", []),
    }
    prompt = SLOT_NARRATIVE_PROMPT + json.dumps(summary_input, ensure_ascii=False)
    messages = [
        {"role": "system", "content": "You are a concise egocentric video summariser."},
        {"role": "user",   "content": prompt},
    ]
    text_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs    = tokenizer(text_prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=300,
            pad_token_id=model.config.pad_token_id,
        )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def _build_day_narrative(daily_digest: Dict, slot_narratives: List[str]) -> str:
    prompt = (
        DAY_NARRATIVE_PROMPT
        + f"\nDAY{daily_digest.get('day')} — "
        + f"{daily_digest.get('total_fios', 0)} interactions\n\n"
        + "\n".join(f"Hour {i+1}: {n}" for i, n in enumerate(slot_narratives) if n)
    )
    messages = [
        {"role": "system", "content": "You are a concise egocentric video summariser."},
        {"role": "user",   "content": prompt},
    ]
    text_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs    = tokenizer(text_prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=500,
            pad_token_id=model.config.pad_token_id,
        )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


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

    job_name = os.environ.get("SLURM_JOB_NAME", "fio_v3_abl_build")
    job_id   = os.environ.get("SLURM_JOB_ID", "")
    exp_name = f"{job_name}_{job_id}" if job_id else job_name
    logger   = start_experiment(exp_name)

    # ── Config ────────────────────────────────────────────────────────────────
    CAPTIONS_PATH     = "/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoGPT/meta_q_guided_7days_jake_05-02-2026-1.json"
    OUTPUT_DIR        = f"./ablation_runs/{ABLATION_TAG}"
    FIO_CACHE_DIR     = os.path.join(CACHE_DIR, "fio_cache")
    RAW_DB_DIR        = os.path.join(OUTPUT_DIR, "chroma_fio_raw")
    OPT_DB_DIR        = os.path.join(OUTPUT_DIR, "chroma_fio_opt")
    HIER_PATH         = os.path.join(OUTPUT_DIR, "hierarchy_memory.json")
    REPORT_PATH       = os.path.join(OUTPUT_DIR, "build_report.json")

    TARGET_DAY        = 7
    TEST_DAY1_ONLY    = False
    SKIP_PHASE1       = True#False
    ENABLE_NARRATIVES = False#True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        raw_db_dir=RAW_DB_DIR, opt_db_dir=OPT_DB_DIR,
        store_raw=True, store_opt=True,
    )
    fio_optimizer = FIOOptimizer(
        sim_threshold=0.75,
        keep_threshold=ABLATION_TAU,            # ◄── from env
        lambda_decay=ABLATION_LAMBDA,           # ◄── from env
        redundancy_cooldown_sec=1.0,
        compression_sim_threshold=0.75,
        alpha_cold_start=0.5, alpha_ema=0.2,
        history_window=5, max_prototypes=200,
        max_action_prototypes=50, max_object_interactions=20,
        max_kept_for_compression=100, max_alpha_clusters=50,
        # Component ablation (Study B)
        disable_frequency=DISABLE_FREQUENCY,
        disable_recency=DISABLE_RECENCY,
        disable_distinctiveness=DISABLE_DISTINCTIVENESS,
        disable_interaction=DISABLE_INTERACTION,
    )
    print(f"[ABLATION] λ={ABLATION_LAMBDA:.3e}  τ={ABLATION_TAU}  tag={ABLATION_TAG}")
    if any([DISABLE_FREQUENCY, DISABLE_RECENCY, DISABLE_DISTINCTIVENESS, DISABLE_INTERACTION]):
        print(f"[ABLATION] Component flags: "
              f"freq={'OFF' if DISABLE_FREQUENCY else 'ON'} "
              f"rec={'OFF' if DISABLE_RECENCY else 'ON'} "
              f"dist={'OFF' if DISABLE_DISTINCTIVENESS else 'ON'} "
              f"inter={'OFF' if DISABLE_INTERACTION else 'ON'}")
    hier_memory = HierarchicalMemory(
        llm_summarise_fn=None, alpha_threshold=0.55, max_raw_per_slot=200,
    )

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

    # ── PHASE 2: Sequential optimizer + hierarchy ─────────────────────────────
    all_kept = run_phase2_optimize(
        captions          = captions,
        cache_dir         = FIO_CACHE_DIR,
        day_number        = TARGET_DAY,
        dual_storage      = dual_storage,
        fio_optimizer     = fio_optimizer,
        hier_memory       = hier_memory,
        tracker           = tracker,
        enable_narratives = ENABLE_NARRATIVES,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    hier_memory.save(HIER_PATH)
    tracker.finish_build(hier_stats=hier_memory.stats, optimizer_stats=fio_optimizer._stats)
    tracker.save_report(REPORT_PATH)
    dual_storage.show_stats()

    # ── Save ablation metadata for downstream eval ──────────────────────────
    ablation_meta = {
        "ablation_tag":   ABLATION_TAG,
        "lambda_decay":   ABLATION_LAMBDA,
        "keep_threshold": ABLATION_TAU,
        "components": {
            "frequency":       not DISABLE_FREQUENCY,
            "recency":         not DISABLE_RECENCY,
            "distinctiveness": not DISABLE_DISTINCTIVENESS,
            "interaction":     not DISABLE_INTERACTION,
        },
        "output_dir":     OUTPUT_DIR,
        "fio_cache":      FIO_CACHE_DIR,
        "target_day":     TARGET_DAY,
        "n_captions":     len(captions),
        "n_kept":         len(all_kept),
    }
    with open(os.path.join(OUTPUT_DIR, "ablation_meta.json"), "w") as f:
        json.dump(ablation_meta, f, indent=2)
    print(f"[INFO] Ablation meta saved → {OUTPUT_DIR}/ablation_meta.json")

    print(f"\n[fio_v3_abl] Complete.")
    print(f"  Kept FIOs : {len(all_kept)}")
    print(f"  FIO cache : {FIO_CACHE_DIR}/")
    print(f"  Hierarchy : {HIER_PATH}")
    print(f"  Report    : {REPORT_PATH}")
    
    #print entries in the FIO
    from fio_count import analyze_chroma_dates
    print(f"\nIn Raw DB:{RAW_DB_DIR}")
    analyze_chroma_dates(RAW_DB_DIR)

    print(f"\nIn Opt DB{OPT_DB_DIR}")
    analyze_chroma_dates(OPT_DB_DIR)

    logger.info("fio_v3_abl build complete")