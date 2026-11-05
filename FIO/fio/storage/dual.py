# fio/storage/dual.py
# ─────────────────────────────────────────────────────────────────────────────
# DualFIOStorage — wraps ChromaFIOStorage to manage two separate DB directories:
#
#   chroma_fio_raw/    — unoptimised FIOs (ablation baseline)
#   chroma_fio_opt/    — optimised FIOs (our system)
#
# Both use identical collection naming: "collection_day1_to_dayN"
# Ablation = swap which storage object you query.
#
# Also provides:
#   get_collection_name(day_n)         → "collection_day1_to_dayN"
#   get_where_filter(query_day)        → date filter dict (fixes Bug C)
#   store_raw_fios(fios, day_n)                 → stores to raw DB
#   store_opt_fios(fios, day_n)                 → stores to opt DB
#   search(query, k, query_day, db="opt")       → routes to correct DB + date filter
#   search_person_filter / search_speech_filter → search with an extra where-filter
#   show_stats() / stats()                      → entry counts per store
#
# DO NOT modify fio/storage/chroma.py — this wraps it unchanged.
# ─────────────────────────────────────────────────────────────────────────────

import json
import re
from typing import Dict, List, Optional, Any

from fio.storage.chroma import ChromaFIOStorage   # existing file — unchanged
from fio.schema import validate_fio, build_document_string


# ══════════════════════════════════════════════════════════════════════════════
# Collection naming
# ══════════════════════════════════════════════════════════════════════════════

def get_collection_name(day_n: int) -> str:
    """
    Cumulative collection for day N.
    A DAY3 question uses collection_day1_to_day3 — contains DAY1+2+3 FIOs.
    This is the correct fix for Bug C (temporal filter regression).
    """
    #return f"collection_day1_to_day{day_n}" # this is ok for fio_build but I have to check for QA
    return "collection_day1_to_day7"


def parse_day_int(day: Any) -> int:
    m = re.search(r"(\d+)", str(day or ""))
    return int(m.group(1)) if m else 1


def get_where_filter(query_day: str) -> Dict:
    """
    Mandatory date filter for every query.
    Prevents DAY1 questions from retrieving DAY6/7 FIOs — fixes Bug C.

    The filter keeps FIOs where date is in [DAY1 .. query_day].
    ChromaDB supports $in for list filtering.
    """
    day_n = parse_day_int(query_day)
    valid_days = [f"DAY{i}" for i in range(1, day_n + 1)]
    return {"date": {"$in": valid_days}}


# ══════════════════════════════════════════════════════════════════════════════
# DualFIOStorage
# ══════════════════════════════════════════════════════════════════════════════

class DualFIOStorage:
    """
    Manages two ChromaFIOStorage instances:
      raw  → chroma_fio_raw/    (unoptimised — ablation baseline)
      opt  → chroma_fio_opt/    (optimised   — our system)

    Both have the same collection structure: collection_day1_to_dayN.
    """

    def __init__(
        self,
        raw_db_dir:  str = "chroma_fio_raw",
        opt_db_dir:  str = "chroma_fio_opt",
        store_raw:   bool = True,    # set False to skip raw DB (save time)
        store_opt:   bool = True,
    ):
        self._store_raw_flag = store_raw
        self._store_opt_flag = store_opt

        if store_raw:
            self.raw_db = ChromaFIOStorage(db_dir=raw_db_dir)
            print(f"[DualFIOStorage] raw DB → {raw_db_dir}/")
        else:
            self.raw_db = None
            print(f"[DualFIOStorage] raw DB disabled")

        if store_opt:
            self.opt_db = ChromaFIOStorage(db_dir=opt_db_dir)
            print(f"[DualFIOStorage] opt DB → {opt_db_dir}/")
        else:
            self.opt_db = None
            print(f"[DualFIOStorage] opt DB disabled")

        # Running counts
        self._raw_stored: int = 0
        self._opt_stored: int = 0

    # ── Store ─────────────────────────────────────────────────────────────────

    def store_raw_fios(self, fios: List[Dict], day_n: int):
        """
        Store unoptimised FIOs into the raw DB.
        Validates fields, adds document string, marks optimized=False.
        """
        if not self._store_raw_flag or not self.raw_db or not fios:
            return

        collection = get_collection_name(day_n) #"collection_day1_to_day{day_n}"
        prepared = []
        for fio in fios:
            f = dict(fio)
            f["optimized"]        = False
            f["importance_score"] = f.get("heuristic_score", f.get("importance_score", 0.0))
            f["recency_score"]    = 0.0   # cannot compute without optimizer context
            f["action_sequence"]  = "[]"
            f["document"]         = build_document_string(f)
            prepared.append(validate_fio(f))

        self.raw_db.m_store_fio_entry(prepared, collection_name=collection)
        self._raw_stored += len(prepared)

    def store_opt_fios(self, fios: List[Dict], day_n: int):
        """
        Store optimised FIOs into the opt DB.
        FIOs should already have importance_score and recency_score set.
        """
        if not self._store_opt_flag or not self.opt_db or not fios:
            return

        collection = get_collection_name(day_n)
        prepared = []
        for fio in fios:
            f = dict(fio)
            f["optimized"] = True
            if "document" not in f or not f["document"]:
                f["document"] = build_document_string(f)
            # Ensure action_sequence is a JSON string (not a list)
            seq = f.get("action_sequence")
            if isinstance(seq, list):
                f["action_sequence"] = json.dumps(seq, ensure_ascii=False)
            elif seq is None or seq == "":
                f["action_sequence"] = "[]"
            # Ensure attributes is a JSON string
            attrs = f.get("attributes")
            if isinstance(attrs, dict):
                f["attributes"] = json.dumps(attrs, ensure_ascii=False)
            prepared.append(validate_fio(f))

        self.opt_db.m_store_fio_entry(prepared, collection_name=collection)
        self._opt_stored += len(prepared)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: Dict,
        k: int,
        query_day: str,
        db: str = "opt",              # "opt" or "raw"
        extra_where: Optional[Dict] = None,
    ) -> Dict:
        """
        Semantic search with mandatory date filter (Bug C fix).

        Args:
            query:       Decomposed question JSON (entity, action, etc.)
            k:           Number of results to retrieve
            query_day:   e.g. "DAY3" — determines collection + date filter
            db:          "opt" (our system) or "raw" (ablation baseline)
            extra_where: Additional ChromaDB where clauses merged in

        Returns:
            ChromaDB search result dict (metadatas, documents, distances)
        """
        day_n      = parse_day_int(query_day)
        collection = get_collection_name(day_n)
        
        
        where      = get_where_filter(query_day)
        #added for debug
        #print("QUERY_TEXT:", repr(query_text))
        print("WHERE:", where)
        #added end

        # Merge extra where clauses
        if extra_where:
            # ChromaDB requires $and for multiple conditions
            where = {"$and": [where, extra_where]}

        storage = self.opt_db if db == "opt" else self.raw_db
        if storage is None:
            raise RuntimeError(f"DB '{db}' is not initialised in DualFIOStorage")

        return storage.semantic_search(
            query=query,
            k=k,
            collection_name=collection,
            where=where,
        )

    def search_person_filter(
        self,
        query: Dict, #{}
        k: int,
        query_day: str,
        db: str = "opt",
    ) -> Dict:
        """
        Search with persons≠"" filter — for RelationMap questions.
        Restricts retrieval to FIOs that have at least one named person.
        """
        return self.search(
            query=query,
            k=k,
            query_day=query_day,
            db=db,
            extra_where={"persons": {"$ne": ""}}, # $ne = "not equal" operator in ChromaDB query language
        )

    def search_speech_filter(
        self,
        query: Dict,
        k: int,
        query_day: str,
        db: str = "opt",
    ) -> Dict:
        """
        Search with speech_content≠"" filter — for HabitInsight SAID and TaskMaster.
        Restricts retrieval to FIOs that contain quoted speech.
        """
        return self.search(
            query=query,
            k=k,
            query_day=query_day,
            db=db,
            extra_where={"speech_content": {"$ne": ""}},
        )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def show_stats(self):
        print(f"\n[DualFIOStorage] raw_stored={self._raw_stored:,} | opt_stored={self._opt_stored:,}")
        if self.raw_db:
            self.raw_db.show_entries()
        if self.opt_db:
            self.opt_db.show_entries()

    @property
    def stats(self) -> Dict:
        return {
            "raw_stored": self._raw_stored,
            "opt_stored": self._opt_stored,
        }