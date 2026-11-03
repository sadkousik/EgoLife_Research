# build_tracker.py
# ─────────────────────────────────────────────────────────────────────────────
# BuildTracker — unified observability for the FIO memory build pipeline.
#
# Tracks across ALL THREE phases of the build:
#   1. FIO extraction (LLM calls via call_qwen)
#   2. FIO optimization (FIOOptimizer.process)
#   3. Hierarchy update (HierarchicalMemory.update)
#
# Zero external dependencies — stdlib only.
# Instantiate once in fio_v2.py, pass reference to DualFIOStorage if needed.
#
# Usage:
#   tracker = BuildTracker(model_name="Qwen2.5-7B-Instruct-local")
#   tracker.start_build(total_captions=6114)
#
#   for caption in captions:
#       tracker.start_caption(caption_idx, day=caption["date"])
#       ...generate_fio_v2(caption, tracker)...
#       tracker.end_caption(n_raw, n_kept, n_compressed, n_dropped)
#
#   tracker.finish_build(hier_stats, optimizer_stats)
#   tracker.save_report("./fio_test/build_report.json")
# ─────────────────────────────────────────────────────────────────────────────

import time
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Any


class BuildTracker:
    """
    Single object that accumulates all timing, token, and error data
    across the entire FIO memory build.
    """

    def __init__(
        self,
        model_name: str = "Qwen2.5-7B-Instruct-local",
        verbose_progress: bool = True,
        eta_window: int = 20,          # rolling window for ETA calculation
    ):
        self.model_name      = model_name
        self.verbose         = verbose_progress
        self.eta_window      = eta_window

        # ── Build-level ───────────────────────────────────────────
        self._t_build_start: float   = 0.0
        self._total_captions: int    = 0

        # ── Per-caption accumulators ──────────────────────────────
        self._caption_idx:    int    = 0
        self._current_day:    str    = ""
        self._t_caption_start: float = 0.0
        self._t_gen_start:    float  = 0.0
        self._t_opt_start:    float  = 0.0
        self._t_hier_start:   float  = 0.0

        # Step-level timing lists
        self._times_gen:  List[float] = []
        self._times_opt:  List[float] = []
        self._times_hier: List[float] = []

        # ── LLM / token stats ─────────────────────────────────────
        self.llm_calls:       int   = 0
        self.llm_input_tok:   int   = 0
        self.llm_output_tok:  int   = 0
        self.llm_latencies:   List[float] = []

        # ── Error stats ───────────────────────────────────────────
        self.json_fail_count:  int   = 0
        self.empty_fio_count:  int   = 0
        self.llm_error_count:  int   = 0
        self._error_log:       List[Dict] = []

        # ── FIO counts ────────────────────────────────────────────
        self.total_raw:        int   = 0
        self.total_kept:       int   = 0
        self.total_compressed: int   = 0
        self.total_dropped:    int   = 0

        # ── Per-day stats ─────────────────────────────────────────
        self._day_stats: Dict[str, Dict] = defaultdict(lambda: {
            "captions": 0, "raw": 0, "kept": 0, "compressed": 0,
            "dropped": 0, "json_fails": 0, "llm_calls": 0,
            "input_tok": 0, "output_tok": 0, "wall_sec": 0.0,
            "t_start": 0.0,
        })
        self._last_day_seen: str = ""

        # ── GPU stats (optional) ──────────────────────────────────
        self.peak_vram_mb: float = 0.0

    # ══════════════════════════════════════════════════════════════
    # Build lifecycle
    # ══════════════════════════════════════════════════════════════

    def start_build(self, total_captions: int):
        self._t_build_start = time.time()
        self._total_captions = total_captions
        print(f"\n{'='*70}")
        print(f"  MEMORY BUILD START — {self.model_name}")
        print(f"  Captions to process: {total_captions:,}")
        print(f"{'='*70}\n")

    def finish_build(
        self,
        hier_stats: Optional[Dict] = None,
        optimizer_stats: Optional[Dict] = None,
    ):
        """Call after all captions processed and hier_memory.finalise() called."""
        self._close_day()   # close the last open day
        total_wall = time.time() - self._t_build_start
        self._print_final_report(total_wall, hier_stats, optimizer_stats)

    # ══════════════════════════════════════════════════════════════
    # Caption lifecycle
    # ══════════════════════════════════════════════════════════════

    def start_caption(self, caption_idx: int, day: str):
        self._caption_idx     = caption_idx
        self._current_day     = day
        self._t_caption_start = time.time()

        # Day boundary tracking
        if day != self._last_day_seen:
            if self._last_day_seen:
                self._close_day()
            self._day_stats[day]["t_start"] = time.time()
            self._last_day_seen = day

        self._day_stats[day]["captions"] += 1

    def start_gen(self):
        """Call immediately before generate_fio_v2() LLM call."""
        self._t_gen_start = time.time()

    def end_gen(
        self,
        input_tokens: int,
        output_tokens: int,
        success: bool,
        error_type: Optional[str] = None,
    ):
        """Call immediately after generate_fio_v2() returns."""
        t_gen = time.time() - self._t_gen_start
        self._times_gen.append(t_gen)

        self.llm_calls       += 1
        self.llm_input_tok   += input_tokens
        self.llm_output_tok  += output_tokens
        self.llm_latencies.append(t_gen)

        d = self._day_stats[self._current_day]
        d["llm_calls"]  += 1
        d["input_tok"]  += input_tokens
        d["output_tok"] += output_tokens

        if not success:
            self.json_fail_count += 1
            d["json_fails"]      += 1
            if error_type:
                self._error_log.append({
                    "caption_idx": self._caption_idx,
                    "day":         self._current_day,
                    "type":        error_type,
                    "t_gen":       round(t_gen, 3),
                })
            if self.llm_calls > 0 and error_type == "llm_none":
                self.llm_error_count += 1

    def start_opt(self):
        self._t_opt_start = time.time()

    def end_opt(self, n_raw: int, n_kept: int, n_compressed: int, n_dropped: int):
        t_opt = time.time() - self._t_opt_start
        self._times_opt.append(t_opt)

        self.total_raw        += n_raw
        self.total_kept       += n_kept
        self.total_compressed += n_compressed
        self.total_dropped    += n_dropped

        d = self._day_stats[self._current_day]
        d["raw"]        += n_raw
        d["kept"]       += n_kept
        d["compressed"] += n_compressed
        d["dropped"]    += n_dropped

    def start_hier(self):
        self._t_hier_start = time.time()

    def end_hier(self):
        t_hier = time.time() - self._t_hier_start
        self._times_hier.append(t_hier)

    def end_caption(self):
        """Call at the very end of each caption loop iteration. Prints progress line."""
        if not self.verbose:
            return
        self._print_progress()

    # ══════════════════════════════════════════════════════════════
    # Slot / day events (called by hierarchy)
    # ══════════════════════════════════════════════════════════════

    def log_slot_built(self, day: int, slot: int, n_fios: int, build_ms: int):
        if self.verbose:
            print(
                f"  [SLOT] DAY{day} slot={slot:02d} | "
                f"{n_fios} FIOs | {build_ms}ms"
            )

    def log_day_built(self, day: int, n_slots: int, n_fios: int, peak_slot: Any):
        wall = time.time() - self._day_stats.get(f"DAY{day}", {}).get("t_start", time.time())
        self._print_day_summary(f"DAY{day}", wall)

    # ══════════════════════════════════════════════════════════════
    # GPU / VRAM
    # ══════════════════════════════════════════════════════════════

    def sample_vram(self):
        """Call periodically (e.g. every 100 captions). No-op if no CUDA."""
        try:
            import torch
            if torch.cuda.is_available():
                mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                self.peak_vram_mb = max(self.peak_vram_mb, mb)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════════

    def save_report(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        report = self._build_report_dict(time.time() - self._t_build_start)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n[BuildTracker] Report saved → {path}")

    # ══════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════

    def _close_day(self):
        d = self._day_stats.get(self._last_day_seen)
        if d and d.get("t_start"):
            d["wall_sec"] = round(time.time() - d["t_start"], 1)

    def _avg(self, lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    def _percentile(self, lst: List[float], p: float) -> float:
        if not lst:
            return 0.0
        s = sorted(lst)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    def _eta_str(self) -> str:
        recent = self._times_gen[-self.eta_window:]
        if not recent:
            return "?"
        avg_gen = self._avg(recent)
        avg_opt = self._avg(self._times_opt[-self.eta_window:]) if self._times_opt else 0.0
        remaining = (self._total_captions - self._caption_idx - 1) * (avg_gen + avg_opt)
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        if mins >= 60:
            return f"{mins//60}h{mins%60:02d}m"
        return f"{mins}m{secs:02d}s"

    def _print_progress(self):
        t_gen  = self._times_gen[-1]  if self._times_gen  else 0.0
        t_opt  = self._times_opt[-1]  if self._times_opt  else 0.0
        t_hier = self._times_hier[-1] if self._times_hier else 0.0

        # Latest FIO counts for this caption
        n_kept = self.total_kept
        n_raw  = self.total_raw

        ok = "✓" if (not self._error_log or
                      self._error_log[-1]["caption_idx"] != self._caption_idx) else "✗"

        tok_this = (self.llm_input_tok + self.llm_output_tok)
        tok_per_call = tok_this // max(self.llm_calls, 1)

        print(
            f"[{self._caption_idx+1:5d}/{self._total_captions} {self._current_day:4s}] "
            f"{ok} "
            f"gen={t_gen:.1f}s opt={t_opt:.2f}s hier={t_hier:.3f}s | "
            f"kept={self.total_kept} raw={self.total_raw} | "
            f"tok≈{tok_per_call} | "
            f"fails={self.json_fail_count} | "
            f"ETA {self._eta_str()}"
        )

    def _print_day_summary(self, day: str, wall_sec: float):
        d = self._day_stats.get(day, {})
        print(f"\n{'━'*70}")
        print(f"  {day} COMPLETE  [wall={_fmt_sec(wall_sec)}]")
        print(f"  Captions:  {d.get('captions',0)} | JSON fails: {d.get('json_fails',0)}")
        print(
            f"  FIOs:      raw={d.get('raw',0)} "
            f"kept={d.get('kept',0)} "
            f"comp={d.get('compressed',0)} "
            f"drop={d.get('dropped',0)}"
        )
        print(
            f"  LLM:       {d.get('llm_calls',0)} calls | "
            f"~{(d.get('input_tok',0)+d.get('output_tok',0))//1000}K tokens"
        )
        print(f"{'━'*70}\n")

    def _print_final_report(
        self,
        total_wall: float,
        hier_stats: Optional[Dict],
        optimizer_stats: Optional[Dict],
    ):
        r = self._build_report_dict(total_wall)
        s = r["summary"]
        t = r["timing"]
        llm = r["llm"]
        fio = r["fio_counts"]
        err = r["errors"]

        print(f"\n{'╔'+'═'*68+'╗'}")
        print(f"║  {'MEMORY BUILD COMPLETE':^66}  ║")
        print(f"╠{'═'*70}╣")

        print(f"║  TIMING{'':<61}║")
        print(f"║    Total wall time:      {_fmt_sec(total_wall):<44}║")
        print(f"║    Avg per caption:      {t['avg_cap_sec']:.2f}s  "
              f"(min {t['min_gen']:.1f}s  max {t['max_gen']:.1f}s){'':<18}║")
        print(f"║    FIO gen (LLM):        {t['avg_gen']:.2f}s avg  ← bottleneck{'':<28}║")
        print(f"║    Optimizer:            {t['avg_opt']:.3f}s avg{'':<42}║")
        print(f"║    Hierarchy update:     {t['avg_hier']:.4f}s avg{'':<41}║")

        print(f"╠{'═'*70}╣")
        print(f"║  FIO STATS{'':<58}║")
        print(f"║    Captions processed:   {fio['captions']:<44}║")
        print(f"║    JSON extract fails:   {err['json_fails']} ({err['json_fail_pct']:.1f}%){'':<40}║")
        print(f"║    Raw FIOs generated:   {fio['total_raw']:<44}║")
        print(f"║    Kept (optimised):     {fio['total_kept']} ({fio['kept_pct']:.1f}%){'':<40}║")
        print(f"║    Compressed:           {fio['total_compressed']:<44}║")
        print(f"║    Dropped:              {fio['total_dropped']:<44}║")

        print(f"╠{'═'*70}╣")
        print(f"║  LLM  ({self.model_name}){'':<44}║")
        print(f"║    Total calls:          {llm['calls']:<44}║")
        print(f"║    Input tokens:         ~{llm['input_tok']//1000}K{'':<44}║")
        print(f"║    Output tokens:        ~{llm['output_tok']//1000}K{'':<44}║")
        print(f"║    Avg latency:          {llm['avg_latency']:.2f}s  "
              f"(p95: {llm['p95_latency']:.1f}s){'':<31}║")
        if self.peak_vram_mb > 0:
            print(f"║    Peak VRAM:            {self.peak_vram_mb:.0f} MB{'':<44}║")

        if hier_stats:
            print(f"╠{'═'*70}╣")
            print(f"║  HIERARCHY{'':<58}║")
            print(f"║    Slot digests built:   {hier_stats.get('slot_digests', '?'):<44}║")
            print(f"║    Daily digests built:  {hier_stats.get('daily_digests', '?'):<44}║")
            print(f"║    LLM calls (hier):     {hier_stats.get('llm_calls', 0):<44}║")

        print(f"╚{'═'*70}╝")

    def _build_report_dict(self, total_wall: float) -> Dict:
        n_caps = max(len(self._times_gen), 1)
        return {
            "model": self.model_name,
            "summary": {
                "total_captions": self._total_captions,
                "total_wall_sec": round(total_wall, 1),
            },
            "timing": {
                "avg_cap_sec":  round(self._avg(self._times_gen) + self._avg(self._times_opt), 3),
                "avg_gen":      round(self._avg(self._times_gen), 3),
                "avg_opt":      round(self._avg(self._times_opt), 3),
                "avg_hier":     round(self._avg(self._times_hier), 4),
                "min_gen":      round(min(self._times_gen) if self._times_gen else 0, 2),
                "max_gen":      round(max(self._times_gen) if self._times_gen else 0, 2),
                "p95_gen":      round(self._percentile(self._times_gen, 95), 2),
            },
            "llm": {
                "calls":        self.llm_calls,
                "input_tok":    self.llm_input_tok,
                "output_tok":   self.llm_output_tok,
                "avg_latency":  round(self._avg(self.llm_latencies), 3),
                "p95_latency":  round(self._percentile(self.llm_latencies, 95), 2),
            },
            "fio_counts": {
                "captions":          n_caps,
                "total_raw":         self.total_raw,
                "total_kept":        self.total_kept,
                "total_compressed":  self.total_compressed,
                "total_dropped":     self.total_dropped,
                "kept_pct":          round(self.total_kept / max(self.total_raw, 1) * 100, 1),
                "comp_pct":          round(self.total_compressed / max(self.total_raw, 1) * 100, 1),
            },
            "errors": {
                "json_fails":       self.json_fail_count,
                "json_fail_pct":    round(self.json_fail_count / max(self.llm_calls, 1) * 100, 2),
                "llm_errors":       self.llm_error_count,
                "error_log":        self._error_log[-20:],   # last 20 errors
            },
            "per_day": dict(self._day_stats),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
        }


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_sec(sec: float) -> str:
    if sec >= 3600:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        return f"{h}h {m:02d}m"
    elif sec >= 60:
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}m {s:02d}s"
    return f"{sec:.1f}s"