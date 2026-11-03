# fio_hierarchy.py
#
# Two-level importance-aware FIO hierarchy.
#
# Boundary logic (corrected):
#   The raw timestamp strings (e.g. "11094208") are NOT wall-clock time —
#   they are session-relative frame counters. Parsing them as HH:MM:SS:FF
#   would put every clip in the same "hour 11" bucket forever.
#
#   Instead:
#     Day boundary  → 'date' field (DAY1, DAY2, ...) — already in every FIO
#     Hour boundary → every CLIPS_PER_SLOT clips processed within a day
#                     Each clip = 30s of video, so 120 clips = 1 actual hour.
#
# Two levels:
#   Level 1 — slot digest    keyed by (day_int, slot_int)
#              one slot = CLIPS_PER_SLOT clips = 1 hour of video
#   Level 2 — daily digest   keyed by day_int
#              one day = all slots within a single 'date' value
#
# Both levels are structured dicts (not prose) so ChromaDB can embed
# and filter them. An optional LLM hook enriches digests with a
# narrative string stored in digest["llm_summary"].
#
# Integration in fio.py:
#
#   hier_memory = HierarchicalMemory()
#
#   for caption in captions:
#       raw_fios   = generate_fio(caption)
#       opt_result = fio_optimizer.process(raw_fios)
#       kept_fios  = opt_result["kept"]
#
#       # pass clip_date so hierarchy knows which day this clip belongs to
#       hier_memory.update(
#           kept_fios,
#           clip_date=caption["date"],
#           alpha_table=opt_result.get("alpha_table", {}),
#       )
#
#   hier_memory.finalise()
#   hier_memory.save("./fio_test/hierarchy.json")
#
#   # query time:
#   ctx = hier_memory.get_context(day="DAY1", slot=3)   # slot 3 = hours 3-4

import re
import json
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple


# ================================================================
# Constants
# ================================================================

CLIPS_PER_SLOT: int = 120   # 120 clips × 30s = 1 hour per slot


# ================================================================
# Helpers
# ================================================================

def _parse_day(day_str: Any) -> int:
    """Extract integer from 'DAY1', 'DAY2', etc."""
    match = re.search(r"(\d+)", str(day_str or ""))
    return int(match.group(1)) if match else 0


def _get_alpha(action: str, alpha_table: Dict) -> float:
    """
    Look up α for an action string from the optimizer's alpha_table output.

    alpha_table is keyed by cluster label (most common action string seen).
    The incoming action may not exactly match the cluster label — e.g. the
    cluster label is 'mention' but the FIO action is 'say'.

    Lookup strategy:
      1. Exact key match (action == cluster label)
      2. action appears in cluster's label_counts (synonym in same cluster)
      3. Falls back to prior (0.5)
    """
    if not action:
        return 0.5
    action_norm = action.lower().strip()

    # 1. exact label match
    entry = alpha_table.get(action_norm)
    if entry and isinstance(entry, dict):
        return float(entry.get("alpha", 0.5))

    # 2. action is a synonym within some cluster's label_counts
    for label, info in alpha_table.items():
        if isinstance(info, dict):
            label_counts = info.get("label_counts", {})
            if action_norm in label_counts:
                return float(info.get("alpha", 0.5))

    return 0.5


# ================================================================
# Digest builders
# ================================================================

def _build_slot_digest(
    fios: List[Dict],
    day: int,
    slot: int,
    alpha_table: Dict,
    alpha_threshold: float,
) -> Dict:
    """
    Build a structured slot digest from a list of kept FIOs.

    slot = clip-count bucket within a day (slot 0 = clips 1-120, slot 1 = 121-240, ...)

    Fields:
      day, slot              — keys
      clip_range             — (first_clip_num, last_clip_num) in this slot
      fio_count              — total kept FIOs in this slot
      avg_importance         — mean importance_score
      top_objects            — up to 5 most frequent non-null objects
      top_actions            — up to 5 most frequent actions
      action_object_pairs    — top 8 (action, object) co-occurrence frequencies
      high_alpha_events      — FIOs whose action α >= alpha_threshold
      attribute_summary      — deduplicated list of "key=value" attribute pairs
      llm_summary            — None by default; filled by LLM hook if enabled
    """
    if not fios:
        return {
            "day": day, "slot": slot,
            "clip_range": (slot * CLIPS_PER_SLOT + 1, (slot + 1) * CLIPS_PER_SLOT),
            "fio_count": 0, "avg_importance": 0.0,
            "top_objects": [], "top_actions": [],
            "action_object_pairs": [], "high_alpha_events": [],
            "attribute_summary": [], "llm_summary": None,
        }

    obj_counts:    Dict[str, int]             = defaultdict(int)
    action_counts: Dict[str, int]             = defaultdict(int)
    pair_counts:   Dict[Tuple[str,str], int]  = defaultdict(int)
    attr_list:     List[str]                  = []
    high_alpha:    List[Dict]                 = []
    importance_sum = 0.0

    for f in fios:
        action = (f.get("action") or "").lower().strip()
        obj    = (f.get("object") or "").lower().strip()
        attrs  = f.get("attributes") or {}
        imp    = float(f.get("importance_score") or f.get("importance_score_old") or 0.0)
        importance_sum += imp

        if action:
            action_counts[action] += 1
        if obj:
            obj_counts[obj] += 1
        if action and obj:
            pair_counts[(action, obj)] += 1

        if isinstance(attrs, dict):
            for k, v in attrs.items():
                if v not in (None, "", [], {}):
                    attr_list.append(f"{k}={v}")

        alpha = _get_alpha(action, alpha_table)
        if alpha >= alpha_threshold:
            high_alpha.append({
                "action":     f.get("action"),
                "object":     f.get("object"),
                "app":        f.get("app"),
                "attributes": attrs,
                "alpha":      round(alpha, 4),
                "importance": round(imp, 4),
                "id":         f.get("id"),
                "start_time": f.get("start_time"),
            })

    n = len(fios)

    # deduplicate high_alpha by FIO id — same FIO can appear across
    # multiple update() calls since kept_output grows cumulatively
    seen_ids: Dict[str, Dict] = {}
    for ev in high_alpha:
        eid = ev.get("id") or f"{ev.get('action')}_{ev.get('object')}_{ev.get('start_time')}"
        if eid not in seen_ids or ev["alpha"] > seen_ids[eid]["alpha"]:
            seen_ids[eid] = ev
    high_alpha = sorted(seen_ids.values(), key=lambda x: (-x["alpha"], -x["importance"]))

    return {
        "day":   day,
        "slot":  slot,
        "clip_range": (slot * CLIPS_PER_SLOT + 1, (slot + 1) * CLIPS_PER_SLOT),
        "fio_count":      n,
        "avg_importance": round(importance_sum / n, 4),
        "top_objects":  [o for o, _ in sorted(obj_counts.items(),    key=lambda x: -x[1])[:5]],
        "top_actions":  [a for a, _ in sorted(action_counts.items(), key=lambda x: -x[1])[:5]],
        "action_object_pairs": [
            {"action": a, "object": o, "count": c}
            for (a, o), c in sorted(pair_counts.items(), key=lambda x: -x[1])[:8]
        ],
        "high_alpha_events":  high_alpha,
        "attribute_summary":  list(set(attr_list))[:20],
        "llm_summary":        None,
    }


def _build_daily_digest(
    slot_digests: List[Dict],
    day: int,
    alpha_table: Dict,
) -> Dict:
    """
    Build a structured daily digest by aggregating slot digests.

    Fields:
      day                    — day int
      total_fios             — sum across all slots
      avg_importance         — weighted mean importance
      total_clips_processed  — total clips seen this day
      total_slots            — number of slot digests
      top_objects            — top 8 objects across the day
      top_actions            — top 8 actions across the day
      peak_slot              — slot index with highest avg_importance
      high_alpha_events      — union of all slot high_alpha_events (top 15, deduped)
      attribute_summary      — union of slot attribute summaries (top 30, deduped)
      slot_breakdown         — per-slot fio_count and avg_importance
      llm_summary            — None by default; filled by LLM hook if enabled
    """
    if not slot_digests:
        return {
            "day": day, "total_fios": 0, "avg_importance": 0.0,
            "total_clips_processed": 0, "total_slots": 0,
            "top_objects": [], "top_actions": [], "peak_slot": None,
            "high_alpha_events": [], "attribute_summary": [],
            "slot_breakdown": [], "llm_summary": None,
        }

    obj_counts:    Dict[str, int] = defaultdict(int)
    action_counts: Dict[str, int] = defaultdict(int)
    all_high_alpha: List[Dict]    = []
    all_attrs:      List[str]     = []
    total_fios     = 0
    importance_sum = 0.0
    slot_breakdown = []

    for d in slot_digests:
        n   = d.get("fio_count", 0)
        imp = d.get("avg_importance", 0.0)
        total_fios     += n
        importance_sum += imp * n

        for obj in d.get("top_objects", []):
            obj_counts[obj] += 1
        for act in d.get("top_actions", []):
            action_counts[act] += 1

        all_high_alpha.extend(d.get("high_alpha_events", []))
        all_attrs.extend(d.get("attribute_summary", []))

        slot_breakdown.append({
            "slot":           d["slot"],
            "clip_range":     d.get("clip_range"),
            "fio_count":      n,
            "avg_importance": imp,
        })

    peak_slot = max(
        slot_breakdown, key=lambda x: x["avg_importance"]
    )["slot"] if slot_breakdown else None

    # deduplicate high_alpha events by id
    seen: Dict[str, Dict] = {}
    for ev in all_high_alpha:
        eid = ev.get("id") or f"{ev.get('action')}_{ev.get('object')}"
        if eid not in seen or ev["alpha"] > seen[eid]["alpha"]:
            seen[eid] = ev
    top_high_alpha = sorted(
        seen.values(), key=lambda x: (-x["alpha"], -x["importance"])
    )[:15]

    return {
        "day":                   day,
        "total_fios":            total_fios,
        "avg_importance":        round(importance_sum / total_fios, 4) if total_fios else 0.0,
        "total_clips_processed": sum(CLIPS_PER_SLOT for _ in slot_digests),
        "total_slots":           len(slot_digests),
        "top_objects":  [o for o, _ in sorted(obj_counts.items(),    key=lambda x: -x[1])[:8]],
        "top_actions":  [a for a, _ in sorted(action_counts.items(), key=lambda x: -x[1])[:8]],
        "peak_slot":             peak_slot,
        "high_alpha_events":     top_high_alpha,
        "attribute_summary":     list(set(all_attrs))[:30],
        "slot_breakdown":        sorted(slot_breakdown, key=lambda x: x["slot"]),
        "llm_summary":           None,
    }


# ================================================================
# HierarchicalMemory
# ================================================================

class HierarchicalMemory:
    """
    Two-level clip-count-aware hierarchy over kept FIOs.

    Boundaries:
      Day   → 'date' field on each FIO / caption (DAY1, DAY2, ...)
      Slot  → every CLIPS_PER_SLOT (default 120) clips within a day
              = 1 actual hour of 30s-clip video

    Level 1  slot digest    — structured aggregate per (day, slot)
    Level 2  daily digest   — structured aggregate per day

    LLM hook (optional):
      Pass llm_summarise_fn = fn(digest: dict, level: str) -> str
      Receives the structured digest and level label ("slot" or "daily").
      Result stored in digest["llm_summary"].
      If None (default), no LLM call is made.

    Example with LLM:
      def my_llm(digest, level):
          prompt = f"Summarise this {level} egocentric activity: {json.dumps(digest)}"
          return call_gpt4o_mini(prompt)

      hier = HierarchicalMemory(llm_summarise_fn=my_llm)
    """

    def __init__(
        self,
        clips_per_slot: int = CLIPS_PER_SLOT,
        llm_summarise_fn: Optional[Callable[[Dict, str], str]] = None,
        alpha_threshold: float = 0.55,
        max_raw_per_slot: int = 200,
    ):
        self._clips_per_slot  = clips_per_slot
        self._llm_fn          = llm_summarise_fn
        self._alpha_threshold = alpha_threshold
        self._max_raw         = max_raw_per_slot

        # clip counter per day — resets when day changes
        self._clip_count_today: int = 0
        self._current_day:      int = -1
        self._current_slot:     int = 0   # slot index within current day

        # raw FIOs per (day_int, slot_int)
        self._raw: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)

        # Level 1: slot digests  keyed by (day_int, slot_int)
        self._slots: Dict[Tuple[int, int], Dict] = {}

        # Level 2: daily digests keyed by day_int
        self._daily: Dict[int, Dict] = {}

        # latest alpha_table from optimizer
        self._alpha_table: Dict = {}

        self.stats = {
            "total_clips":    0,
            "total_fios":     0,
            "slot_digests":   0,
            "daily_digests":  0,
            "llm_calls":      0,
        }

    # ── LLM hook ────────────────────────────────────────────────

    def _maybe_llm(self, digest: Dict, level: str) -> Dict:
        if self._llm_fn is None:
            return digest
        try:
            digest["llm_summary"] = self._llm_fn(digest, level)
            self.stats["llm_calls"] += 1
        except Exception as e:
            print(f"[HierarchicalMemory] LLM call failed ({e}) — skipping")
        return digest

    # ── Slot finalisation ────────────────────────────────────────

    def _finalise_slot(self, day: int, slot: int):
        """Build and store a Level 1 slot digest."""
        key   = (day, slot)
        fios  = self._raw.get(key, [])
        digest = _build_slot_digest(
            fios, day, slot, self._alpha_table, self._alpha_threshold
        )
        digest = self._maybe_llm(digest, "slot")
        self._slots[key] = digest
        self.stats["slot_digests"] += 1
        print(
            f"[HierarchicalMemory] Slot digest: DAY{day} slot={slot} "
            f"(clips {digest['clip_range'][0]}–{digest['clip_range'][1]} | "
            f"{len(fios)} FIOs | avg_imp={digest['avg_importance']} | "
            f"high_α={len(digest['high_alpha_events'])})"
        )

    def _finalise_day(self, day: int):
        """Build and store a Level 2 daily digest from all slot digests for this day."""
        slot_digs = [
            digest for (d, s), digest in self._slots.items()
            if d == day
        ]
        digest = _build_daily_digest(slot_digs, day, self._alpha_table)
        digest = self._maybe_llm(digest, "daily")
        self._daily[day] = digest
        self.stats["daily_digests"] += 1
        print(
            f"[HierarchicalMemory] Daily digest: DAY{day} "
            f"({digest['total_fios']} FIOs | {digest['total_slots']} slots | "
            f"peak_slot={digest['peak_slot']} | "
            f"top_objects={digest['top_objects'][:3]})"
        )

    # ── Public API ───────────────────────────────────────────────

    def update(
        self,
        kept_fios: List[Dict],
        clip_date: str,
        alpha_table: Optional[Dict] = None,
    ):
        """
        Ingest kept FIOs from ONE clip and advance the clip counter.

        Call once per caption/clip processed (even if kept_fios is empty —
        the clip counter must advance regardless of filtering outcome).

        Args:
            kept_fios:   result["kept"] from FIOOptimizer.process()
                         May be empty if all FIOs were filtered out.
            clip_date:   caption["date"] — e.g. "DAY1"
                         Used to detect day boundaries.
            alpha_table: result["alpha_table"] from FIOOptimizer.process()
        """
        if alpha_table is not None:
            self._alpha_table = alpha_table

        day = _parse_day(clip_date)

        # ── Day boundary ─────────────────────────────────────────
        if day != self._current_day:
            if self._current_day != -1:
                # finalise the last open slot of the previous day
                self._finalise_slot(self._current_day, self._current_slot)
                # finalise the previous day
                self._finalise_day(self._current_day)
            # reset counters for new day
            self._current_day       = day
            self._clip_count_today  = 0
            self._current_slot      = 0

        # ── Advance clip counter ─────────────────────────────────
        self._clip_count_today += 1
        self.stats["total_clips"] += 1

        # ── Slot boundary ────────────────────────────────────────
        # slot index = (clip_count_today - 1) // clips_per_slot
        new_slot = (self._clip_count_today - 1) // self._clips_per_slot

        if new_slot != self._current_slot:
            # crossing into a new slot — finalise the completed one
            self._finalise_slot(self._current_day, self._current_slot)
            self._current_slot = new_slot

        # ── Store FIOs into current slot ─────────────────────────
        slot_key = (day, self._current_slot)
        bucket   = self._raw[slot_key]
        for fio in kept_fios:
            if len(bucket) < self._max_raw:
                bucket.append(fio)
            self.stats["total_fios"] += 1

    def finalise(self):
        """
        Flush the last open slot and day into digests.
        Call once after all clips have been processed.
        """
        if self._current_day == -1:
            print("[HierarchicalMemory] Nothing to finalise.")
            return

        # finalise last open slot
        last_key = (self._current_day, self._current_slot)
        if last_key not in self._slots:
            self._finalise_slot(self._current_day, self._current_slot)

        # finalise last day
        if self._current_day not in self._daily:
            self._finalise_day(self._current_day)

        print(f"[HierarchicalMemory] Finalised — {self.stats}")

    def get_context(
        self,
        day: Optional[str] = None,
        slot: Optional[int] = None,
        levels: List[int] = [1, 2],
        max_raw: int = 20,
        min_importance: float = 0.0,
        min_alpha: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Retrieve hierarchy context.

        Args:
            day:            e.g. "DAY1" or None for all days
            slot:           slot index (0-based) or None for all slots
            levels:         [0]=raw FIOs, [1]=slot digests, [2]=daily digests
            max_raw:        cap on raw FIOs returned
            min_importance: filter raw FIOs below this score
            min_alpha:      filter high_alpha_events below this α

        Returns:
            {"raw": [...], "slots": {...}, "daily": {...}}
        """
        day_int  = _parse_day(day) if day  is not None else None
        slot_int = int(slot)       if slot is not None else None
        ctx: Dict[str, Any] = {}

        if 0 in levels:
            raw = []
            for (d, s), fios in self._raw.items():
                if day_int  is not None and d != day_int:  continue
                if slot_int is not None and s != slot_int: continue
                for f in fios:
                    imp = float(f.get("importance_score") or f.get("importance_score_old") or 0.0)
                    if imp >= min_importance:
                        raw.append(f)
            ctx["raw"] = raw[:max_raw]

        if 1 in levels:
            slots = {}
            for (d, s), digest in self._slots.items():
                if day_int  is not None and d != day_int:  continue
                if slot_int is not None and s != slot_int: continue
                out = dict(digest)
                if min_alpha > 0:
                    out["high_alpha_events"] = [
                        e for e in out.get("high_alpha_events", [])
                        if e.get("alpha", 0) >= min_alpha
                    ]
                slots[f"DAY{d}_slot{s}"] = out
            ctx["slots"] = slots

        if 2 in levels:
            daily = {}
            for d, digest in self._daily.items():
                if day_int is not None and d != day_int: continue
                out = dict(digest)
                if min_alpha > 0:
                    out["high_alpha_events"] = [
                        e for e in out.get("high_alpha_events", [])
                        if e.get("alpha", 0) >= min_alpha
                    ]
                daily[f"DAY{d}"] = out
            ctx["daily"] = daily

        return ctx

    def get_high_alpha_events(
        self,
        day: Optional[str] = None,
        min_alpha: float = 0.55,
        top_k: int = 20,
    ) -> List[Dict]:
        """
        Return highest-α events across all slot digests for a day.
        Useful for HabitInsight and TaskMaster QA retrieval.
        """
        day_int = _parse_day(day) if day is not None else None
        events  = []
        for (d, s), digest in self._slots.items():
            if day_int is not None and d != day_int:
                continue
            for ev in digest.get("high_alpha_events", []):
                if ev.get("alpha", 0) >= min_alpha:
                    ev_copy = dict(ev)
                    ev_copy["day"]  = d
                    ev_copy["slot"] = s
                    events.append(ev_copy)
        return sorted(
            events, key=lambda x: (-x["alpha"], -x["importance"])
        )[:top_k]

    def to_dict(self) -> Dict:
        return {
            "slots": {
                f"{d}_{s}": digest
                for (d, s), digest in self._slots.items()
            },
            "daily": {
                str(d): digest
                for d, digest in self._daily.items()
            },
            "stats": self.stats,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"[HierarchicalMemory] Saved → {path}")

    def __repr__(self) -> str:
        return (
            f"HierarchicalMemory("
            f"clips_per_slot={self._clips_per_slot} | "
            f"slots={len(self._slots)} | "
            f"daily={len(self._daily)} | "
            f"clips={self.stats['total_clips']} | "
            f"fios={self.stats['total_fios']} | "
            f"llm_calls={self.stats['llm_calls']})"
        )