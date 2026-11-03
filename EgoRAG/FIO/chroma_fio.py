# Complete Modified `chroma_fio.py`
# chroma_fio.py
# ─────────────────────────────────────────────────────────────────────────────
# Modernized ChromaFIOStorage for FIO v2
#
# Compatible with:
#   - fio_v2.py
#   - chroma_fio_v2.py
#   - fio_schema.py
#
# Major fixes:
#   ✓ persons replaces person
#   ✓ speech_content embedded
#   ✓ source/target locations
#   ✓ where filter support
#   ✓ dynamic collection correctness
#   ✓ schema-driven validation
#   ✓ proper duplicate filtering
#   ✓ optimizer metadata support
#   ✓ action_sequence persistence
#   ✓ alpha_score support
#   ✓ optimized flag support
#   ✓ Bug C temporal leakage fix
# ─────────────────────────────────────────────────────────────────────────────

import math
from typing import Dict, Any, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from fio_schema import (
    validate_fio,
    build_document_string,
    CHROMADB_METADATA_FIELDS,
)


class ChromaFIOStorage_not_used:
    def __init__(
        self,
        db_dir: str = "./chroma_fio_db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        print(f"[ChromaFIOStorage] Initializing DB → {db_dir}")

        self.client = chromadb.PersistentClient(path=db_dir)

        self.db_dir = db_dir          # ← ADD: needed to reconnect later

        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # default collection fallback
        self.collection = self.client.get_or_create_collection(
            name="fio_entries",
            embedding_function=self.embedding_function,
        )

    #added on 12062026 
    #---------------------for reconnect-------------------------------------
    def _reconnect(self):
        """Drop the poisoned client/connection and build a fresh one."""
        print(f"[INFO] reconnecting Chroma client → {self.db_dir}")
        # ChromaDB caches clients per path, so a plain PersistentClient() may hand
        # back the SAME poisoned system. Clear the cache first to force a fresh one.
        for path in ("chromadb.api.shared_system_client", "chromadb.api.client"):
            try:
                mod = __import__(path, fromlist=["SharedSystemClient"])
                mod.SharedSystemClient.clear_system_cache()
                break
            except Exception:
                continue
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(
            name="fio_entries",
            embedding_function=self.embedding_function,
        )

    def _resilient_query(self, collection_name, query_text, n_results, where, retries=1):
        """get_collection + query, with one reconnect-and-retry on DB errors."""
        last_err = None
        for attempt in range(retries + 1):
            try:
                collection = self._get_collection(collection_name)
                return collection.query(
                    query_texts=[str(query_text)],
                    n_results=int(n_results),
                    include=["documents", "metadatas", "distances"],
                    where=where,
                )
            except Exception as e:
                last_err = e
                print(f"[WARN] query attempt {attempt} failed: {type(e).__name__}: {e}")
                if attempt < retries:
                    self._reconnect()
        print(f"[WARN] all attempts failed ({type(last_err).__name__}) — no results.")
        return None
    #---------------------for reconnect end---------------------------------
    
    # ═══════════════════════════════════════════════════════════════════════
    # Utility helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _build_query_text(self, query: Dict[str, Any]) -> str:
        """
        Build semantic query string.

        Supports:
        - entity
        - action
        - related_person
        - object_relation
        - speech_content
        - query_intent
        """

        parts = [
            str(query.get("action", "") or ""),
            str(query.get("entity", "") or ""),
            str(query.get("related_person", "") or ""),
            str(query.get("object_relation", "") or ""),
            str(query.get("speech_content", "") or ""),
            str(query.get("query_intent", "") or ""),
        ]

        return " ".join(p.strip() for p in parts if p and str(p).strip())

    # ═══════════════════════════════════════════════════════════════════════
    # Temporal reranking
    # ═══════════════════════════════════════════════════════════════════════

    def _temporal_preference_score(
        self,
        recency_score: float,
        temporal_relation: Any,
    ) -> float:
        relation = str(temporal_relation or "").lower().strip()

        if relation in {"last_time", "today", "after", "recent"}:
            return recency_score

        if relation in {"first_time", "before", "earliest"}:
            return 1.0 - recency_score

        return recency_score

    def _rerank_results(
        self,
        query: Dict[str, Any],
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        distances: List[Any],
        k: int,
        weights: Dict[str, float],
    ) -> Dict[str, Any]:

        temporal_relation = query.get("temporal_relation")

        ranked_rows = []

        for idx, entry_id in enumerate(ids):
            metadata = dict(metadatas[idx] or {})

            distance = distances[idx] if idx < len(distances) else 1.0

            semantic_score = 1.0 / (1.0 + max(self._safe_float(distance, 1.0), 0.0))

            recency_score = self._safe_float(
                metadata.get("recency_score", 0.0)
            )

            importance_score = self._safe_float(
                metadata.get("importance_score", 0.0)
            )

            alpha_score = self._safe_float(
                metadata.get("alpha_score", 0.5)
            )

            temporal_score = self._temporal_preference_score(
                recency_score,
                temporal_relation,
            )

            final_score = (
                weights["semantic"] * semantic_score
                + weights["recency"] * temporal_score
                + weights["importance"] * importance_score
                + weights.get("alpha", 0.05) * alpha_score
            )

            metadata["semantic_score"] = round(semantic_score, 4)
            metadata["retrieval_score"] = round(final_score, 4)

            ranked_rows.append(
                {
                    "id": entry_id,
                    "document": documents[idx] if idx < len(documents) else "",
                    "metadata": metadata,
                    "distance": distance,
                    "score": round(final_score, 4),
                }
            )

        ranked_rows.sort(key=lambda x: x["score"], reverse=True)
        ranked_rows = ranked_rows[:k]

        return {
            "ids": [[r["id"] for r in ranked_rows]],
            "documents": [[r["document"] for r in ranked_rows]],
            "metadatas": [[r["metadata"] for r in ranked_rows]],
            "distances": [[r["distance"] for r in ranked_rows]],
            "scores": [[r["score"] for r in ranked_rows]],
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Collection helper
    # ═══════════════════════════════════════════════════════════════════════

    def _get_collection(self, collection_name: Optional[str] = None):
        if collection_name:
            return self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        print(f"[INFO] collection get/create{' → ' + collection_name if collection_name else ''}")
        return self.collection

    # ═══════════════════════════════════════════════════════════════════════
    # Storage
    # ═══════════════════════════════════════════════════════════════════════

    def m_store_fio_entry(
        self,
        fio_entries: List[Dict[str, Any]],
        batch_size: int = 64,
        skip_existing: bool = True,
        collection_name: Optional[str] = None,
    ):
        """
        Batch store validated FIO entries.
        """

        if not fio_entries:
            print("[INFO] No FIO entries to store.")
            return

        collection = self._get_collection(collection_name)

        # ──────────────────────────────────────────────────────────────
        # Validate + normalize all entries
        # ──────────────────────────────────────────────────────────────

        cleaned_entries = []

        for fio in fio_entries:
            cleaned = validate_fio(fio)

            # always regenerate embedding text
            cleaned["document"] = build_document_string(cleaned)

            cleaned_entries.append(cleaned)

        fio_entries = cleaned_entries

        print(f"[INFO] Validated {len(fio_entries)} FIO entries.")

        # ──────────────────────────────────────────────────────────────
        # Duplicate filtering
        # ──────────────────────────────────────────────────────────────

        if skip_existing:
            try:
                existing = collection.get(include=[])
                existing_ids = set(existing.get("ids", []))
            except Exception:
                existing_ids = set()

            before = len(fio_entries)

            fio_entries = [
                fio for fio in fio_entries
                if fio["id"] not in existing_ids
            ]

            skipped = before - len(fio_entries)

            print(
                f"[INFO] Duplicate filtering | "
                f"Skipped={skipped} | "
                f"Remaining={len(fio_entries)}"
            )

            if not fio_entries:
                print("[INFO] Nothing new to insert.")
                return

        # ──────────────────────────────────────────────────────────────
        # Batch insert
        # ──────────────────────────────────────────────────────────────

        docs_batch = []
        metas_batch = []
        ids_batch = []

        total_inserted = 0

        for fio in fio_entries:
            document = fio["document"]

            metadata = {
                field: fio.get(field)
                for field in CHROMADB_METADATA_FIELDS
                if field != "document"
            }

            docs_batch.append(document)
            metas_batch.append(metadata)
            ids_batch.append(fio["id"])

            if len(docs_batch) >= batch_size:
                collection.upsert(
                    documents=docs_batch,
                    metadatas=metas_batch,
                    ids=ids_batch,
                )

                total_inserted += len(docs_batch)

                print(f"[INFO] Inserted batch | total={total_inserted}")

                docs_batch.clear()
                metas_batch.clear()
                ids_batch.clear()

        # flush remaining
        if docs_batch:
            collection.upsert(
                documents=docs_batch,
                metadatas=metas_batch,
                ids=ids_batch,
            )

            total_inserted += len(docs_batch)

        print(
            f"[SUCCESS] Inserted {total_inserted} FIO entries into ChromaDB"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Semantic search
    # ═══════════════════════════════════════════════════════════════════════

    def semantic_search_old(
        self,
        query: Dict[str, Any],
        k: int = 5,
        candidate_pool: int = 30,
        weights: Optional[Dict[str, float]] = None,
        collection_name: Optional[str] = None,
        where: Optional[Dict] = None,
    ) -> Dict:
        """
        Semantic retrieval with:
        - temporal filtering
        - reranking
        - alpha support
        - recency support
        - importance support
        """

        weights = weights or {
            "semantic": 0.55,
            "recency": 0.20,
            "importance": 0.20,
            "alpha": 0.05,
        }
        
        query_text = self._build_query_text(query)
        #added for debug
        if not query_text.strip():
            print("EMPTY QUERY")
        #end added for debug

        collection = self._get_collection(collection_name)

        raw_results = collection.query(
            query_texts=[query_text],
            n_results=max(k, candidate_pool),
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        if not ids:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "scores": [[]],
            }

        return self._rerank_results(
            query=query,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            distances=distances,
            k=k,
            weights=weights,
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Drop-in replacement for ChromaFIOStorage.semantic_search
    # (replaces lines ~384–570 in chroma_fio.py)
    #
    # CHANGE vs original: on ANY DB read error (e.g. sqlite3.DatabaseError
    # "database disk image is malformed"), this no longer `raise`s — it returns
    # an empty result so the QA loop treats it as "no FIOs retrieved", cascades /
    # answers "Cannot be determined", and CONTINUES to the next question instead
    # of crashing the whole run.
    #
    # ⚠ NOTE: a run that skips corrupt partitions yields DAY1-real + everything-
    # else-empty results. That accuracy is for code testing only — NOT reportable.
    # The real fix is a clean rebuild of the store.
    # ═══════════════════════════════════════════════════════════════════════
    def semantic_search_old1(
        self,
        query: Dict[str, Any],
        k: int = 5,
        candidate_pool: int = 30,
        weights: Optional[Dict[str, float]] = None,
        collection_name: Optional[str] = None,
        where: Optional[Dict] = None,
    ) -> Dict:
        """
        Semantic retrieval with temporal filtering, reranking, and
        alpha / recency / importance support.

        Hardened: any DB-level read failure returns an empty result set
        rather than raising, so a corrupt partition cannot abort the eval.
        """
        # Single source of truth for the "no results" shape.
        EMPTY = {
            "ids":       [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "scores":    [[]],
        }

        weights = weights or {
            "semantic":   0.55,
            "recency":    0.20,
            "importance": 0.20,
            "alpha":      0.05,
        }

        query_text = self._build_query_text(query)
        collection = self._get_collection(collection_name)

        # --------------------------------------------------------------
        # Count guard — skip (don't crash) on DB read error
        # --------------------------------------------------------------
        try:
            collection_count = collection.count()
            print("COUNT OK:", collection_count)
        except Exception as e:
            print(f"[WARN] COUNT FAILED on '{collection.name}': {e}")
            print("[WARN] DB read error — skipping this query, returning no results.")
            return EMPTY

        print(
            f"[INFO] semantic_search"
            f" → collection={collection_name}"
            f" | total_docs={collection_count}"
        )

        if collection_count == 0:
            print("[WARN] Empty collection. Returning no results.")
            return EMPTY

        # --------------------------------------------------------------
        # Prevent Chroma HNSW runtime errors
        # --------------------------------------------------------------
        requested_results = max(k, candidate_pool)
        n_results = min(requested_results, collection_count)   # never exceed available
        n_results = max(1, n_results)

        print(
            f"[INFO] query settings"
            f" | requested={requested_results}"
            f" | final_n_results={n_results}"
            f" | k={k}"
            f" | candidate_pool={candidate_pool}"
        )
        #debug
        # ▼▼▼ ADD THIS ▼▼▼
        print("DEBUG bind types | n_results:", type(n_results),
              "| where vals:", [type(v) for v in
                                (where.get('date', {}).get('$in', []) if where else [])],
              "| qtext:", type(query_text))
        # ▲▲▲ ADD THIS ▲▲▲
        # --------------------------------------------------------------
        # Main query — skip (don't crash) on DB read error
        # --------------------------------------------------------------
        try:
            raw_results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
                where=where,
                #where=None,
            )
        except Exception as e:
            print(f"[WARN] CHROMA QUERY FAILED on '{collection.name}': "
                  f"{type(e).__name__}: {e}")
            print(f"       where={where} | query_text={query_text!r}")
            print("[WARN] DB read error — skipping this query, returning no results.")
            return EMPTY

        # --------------------------------------------------------------
        # Extract results safely
        # --------------------------------------------------------------
        ids       = raw_results.get("ids",       [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        if not ids:
            print("[INFO] No matching results found.")
            return EMPTY

        print(f"[INFO] Retrieved {len(ids)} candidates")

        # --------------------------------------------------------------
        # Rerank final results
        # --------------------------------------------------------------
        return self._rerank_results(
            query=query,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            distances=distances,
            k=min(k, len(ids)),
            weights=weights,
        )
    
    #semantic search new added on 1206
    # ═══════════════════════════════════════════════════════════════════════
    # Drop-in replacement for ChromaFIOStorage.semantic_search
    #
    # PREREQUISITE: add `self.db_dir = db_dir` in __init__ (right after
    #   `self.client = chromadb.PersistentClient(path=db_dir)`), so the
    #   reconnect logic below knows which path to re-open.
    #
    # BEHAVIOR:
    #   - On any DB read error (malformed / InterfaceError / etc.), it throws out
    #     the poisoned Chroma client, clears Chroma's cached system so a genuinely
    #     fresh connection is built, re-opens the collection, and retries ONCE.
    #   - If the retry also fails, it returns an empty result so the QA loop
    #     cascades / answers "Cannot be determined" and continues to the next
    #     question instead of crashing.
    #
    # NOTE: self-healing fixes the stale-connection / NFS-cache failure mode.
    #   It cannot fix a genuinely corrupt store — for that, rebuild on local disk.
    # ═══════════════════════════════════════════════════════════════════════
    def semantic_search(
        self,
        query: Dict[str, Any],
        k: int = 5,
        candidate_pool: int = 30,
        weights: Optional[Dict[str, float]] = None,
        collection_name: Optional[str] = None,
        where: Optional[Dict] = None,
    ) -> Dict:
        """
        Semantic retrieval with temporal filtering, reranking, and
        alpha / recency / importance support.

        Hardened: a poisoned connection is reconnected and retried once; any
        unrecoverable DB read returns an empty result set instead of raising.
        """
        # Single source of truth for the "no results" shape.
        EMPTY = {
            "ids":       [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "scores":    [[]],
        }

        weights = weights or {
            "semantic":   0.55,
            "recency":    0.20,
            "importance": 0.20,
            "alpha":      0.05,
        }

        query_text = self._build_query_text(query)

        # ------------------------------------------------------------------
        # Nested reconnect: drop the poisoned client and build a fresh one.
        # ChromaDB caches clients per path, so a plain PersistentClient() may
        # return the SAME poisoned system — clear the cache first.
        # ------------------------------------------------------------------
        def _reset_client():
            print(f"[INFO] reconnecting Chroma client → {getattr(self, 'db_dir', '?')}")
            for mod_path in ("chromadb.api.shared_system_client",
                             "chromadb.api.client"):
                try:
                    mod = __import__(mod_path, fromlist=["SharedSystemClient"])
                    mod.SharedSystemClient.clear_system_cache()
                    break
                except Exception:
                    continue
            self.client = chromadb.PersistentClient(path=self.db_dir)
            self.collection = self.client.get_or_create_collection(
                name="fio_entries",
                embedding_function=self.embedding_function,
            )

        # ------------------------------------------------------------------
        # Resilient read: get_collection -> count -> query, reconnect+retry once.
        # ------------------------------------------------------------------
        RETRIES = 1
        raw_results = None
        for attempt in range(RETRIES + 1):
            try:
                collection = self._get_collection(collection_name)

                collection_count = collection.count()
                print("COUNT OK:", collection_count)
                print(
                    f"[INFO] semantic_search"
                    f" → collection={collection_name}"
                    f" | total_docs={collection_count}"
                )

                if collection_count == 0:
                    print("[WARN] Empty collection. Returning no results.")
                    return EMPTY

                # Prevent Chroma HNSW runtime errors: never exceed available docs.
                requested_results = max(k, candidate_pool)
                n_results = max(1, min(requested_results, collection_count))

                print(
                    f"[INFO] query settings"
                    f" | requested={requested_results}"
                    f" | final_n_results={n_results}"
                    f" | k={k}"
                    f" | candidate_pool={candidate_pool}"
                )

                raw_results = collection.query(
                    query_texts=[str(query_text)],
                    n_results=int(n_results),
                    include=["documents", "metadatas", "distances"],
                    where=where,
                )
                break  # success — leave the retry loop

            except Exception as e:
                print(f"[WARN] DB read failed (attempt {attempt}) on "
                      f"'{collection_name}': {type(e).__name__}: {e}")
                print(f"       where={where} | query_text={query_text!r}")
                if attempt < RETRIES:
                    print("[INFO] reconnecting client and retrying once...")
                    _reset_client()
                else:
                    print("[WARN] retry exhausted — skipping this query, "
                          "returning no results.")
                    return EMPTY

        # ------------------------------------------------------------------
        # Extract results safely
        # ------------------------------------------------------------------
        ids       = raw_results.get("ids",       [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        if not ids:
            print("[INFO] No matching results found.")
            return EMPTY

        print(f"[INFO] Retrieved {len(ids)} candidates")

        # ------------------------------------------------------------------
        # Rerank final results
        # ------------------------------------------------------------------
        return self._rerank_results(
            query=query,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            distances=distances,
            k=min(k, len(ids)),
            weights=weights,
        )
    #end semantic search 12062026
    
    
    # ═══════════════════════════════════════════════════════════════════════
    # Debug helpers
    # ═══════════════════════════════════════════════════════════════════════

    def show_entries(
        self,
        n: int = 5,
        collection_name: Optional[str] = None,
    ):
        collection = self._get_collection(collection_name)

        data = collection.get(include=["documents", "metadatas"])

        total = len(data.get("ids", []))

        print(f"\n[ChromaFIOStorage] Total entries: {total}")
        print("-" * 80)

        for i in range(min(n, total)):
            print(f"Entry {i+1}")
            print(f"ID:        {data['ids'][i]}")
            print(f"Document:  {data['documents'][i]}")
            print(f"Metadata:  {data['metadatas'][i]}")
            print("-" * 80)

#updated
class ChromaFIOStorage:
    def __init__(
        self,
        db_dir: str = "./chroma_fio_db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        print(f"[ChromaFIOStorage] Initializing DB → {db_dir}")

        self.db_dir = db_dir                      # needed to reconnect later
        self.embedding_model = embedding_model

        self.client = chromadb.PersistentClient(path=db_dir)

        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # default collection fallback
        self.collection = self.client.get_or_create_collection(
            name="fio_entries",
            embedding_function=self.embedding_function,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Reconnect / resilience  (added 12-06-2026)
    # ═══════════════════════════════════════════════════════════════════════
    def _reconnect(self):
        """
        Close the poisoned client/connection, then build a fresh one.

        Used to recover from stale-connection / NFS-cache errors (sqlite3
        'database disk image is malformed', InterfaceError, etc.) that a
        long-lived connection can throw even when the file itself reads fine
        from a fresh process.
        """
        print(f"[INFO] reconnecting Chroma client → {self.db_dir}")

        # 1) Close the existing connection so its sqlite handle / vector
        #    segments are released before we open a new one.
        try:
            self.client._system.stop()          # stops DB + segments
        except Exception as e:
            print(f"[INFO] old-client stop skipped: {type(e).__name__}: {e}")
        self.client = None

        # 2) Clear Chroma's per-path system cache, else PersistentClient() may
        #    hand back the SAME (now-stopped) system instead of a fresh one.
        for mod_path in ("chromadb.api.shared_system_client",
                         "chromadb.api.client"):
            try:
                mod = __import__(mod_path, fromlist=["SharedSystemClient"])
                mod.SharedSystemClient.clear_system_cache()
                break
            except Exception:
                continue

        # 3) Open a fresh client + default collection.
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(
            name="fio_entries",
            embedding_function=self.embedding_function,
        )

    def _resilient_query(self, collection_name, query_text, n_results, where, retries=1):
        """get_collection + query, with one close-reconnect-and-retry on DB errors."""
        last_err = None
        for attempt in range(retries + 1):
            try:
                collection = self._get_collection(collection_name)
                return collection.query(
                    query_texts=[str(query_text)],
                    n_results=int(n_results),
                    include=["documents", "metadatas", "distances"],
                    where=where,
                )
            except Exception as e:
                last_err = e
                print(f"[WARN] query attempt {attempt} failed: "
                      f"{type(e).__name__}: {e}")
                if attempt < retries:
                    self._reconnect()
        print(f"[WARN] all attempts failed ({type(last_err).__name__}) — no results.")
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Utility helpers
    # ═══════════════════════════════════════════════════════════════════════
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _build_query_text(self, query: Dict[str, Any]) -> str:
        """Build semantic query string from the decomposed query fields."""
        parts = [
            str(query.get("action", "") or ""),
            str(query.get("entity", "") or ""),
            str(query.get("related_person", "") or ""),
            str(query.get("object_relation", "") or ""),
            str(query.get("speech_content", "") or ""),
            str(query.get("query_intent", "") or ""),
        ]
        return " ".join(p.strip() for p in parts if p and str(p).strip())

    # ═══════════════════════════════════════════════════════════════════════
    # Temporal reranking
    # ═══════════════════════════════════════════════════════════════════════
    def _temporal_preference_score(
        self,
        recency_score: float,
        temporal_relation: Any,
    ) -> float:
        relation = str(temporal_relation or "").lower().strip()
        if relation in {"last_time", "today", "after", "recent"}:
            return recency_score
        if relation in {"first_time", "before", "earliest"}:
            return 1.0 - recency_score
        return recency_score

    def _rerank_results(
        self,
        query: Dict[str, Any],
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        distances: List[Any],
        k: int,
        weights: Dict[str, float],
    ) -> Dict[str, Any]:

        temporal_relation = query.get("temporal_relation")
        ranked_rows = []

        for idx, entry_id in enumerate(ids):
            metadata = dict(metadatas[idx] or {})
            distance = distances[idx] if idx < len(distances) else 1.0

            semantic_score = 1.0 / (1.0 + max(self._safe_float(distance, 1.0), 0.0))
            recency_score = self._safe_float(metadata.get("recency_score", 0.0))
            importance_score = self._safe_float(metadata.get("importance_score", 0.0))
            alpha_score = self._safe_float(metadata.get("alpha_score", 0.5))

            temporal_score = self._temporal_preference_score(
                recency_score, temporal_relation
            )

            final_score = (
                weights["semantic"] * semantic_score
                + weights["recency"] * temporal_score
                + weights["importance"] * importance_score
                + weights.get("alpha", 0.05) * alpha_score
            )

            metadata["semantic_score"] = round(semantic_score, 4)
            metadata["retrieval_score"] = round(final_score, 4)

            ranked_rows.append({
                "id": entry_id,
                "document": documents[idx] if idx < len(documents) else "",
                "metadata": metadata,
                "distance": distance,
                "score": round(final_score, 4),
            })

        ranked_rows.sort(key=lambda x: x["score"], reverse=True)
        ranked_rows = ranked_rows[:k]

        return {
            "ids":       [[r["id"] for r in ranked_rows]],
            "documents": [[r["document"] for r in ranked_rows]],
            "metadatas": [[r["metadata"] for r in ranked_rows]],
            "distances": [[r["distance"] for r in ranked_rows]],
            "scores":    [[r["score"] for r in ranked_rows]],
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Collection helper
    # ═══════════════════════════════════════════════════════════════════════
    def _get_collection(self, collection_name: Optional[str] = None):
        if collection_name:
            return self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        print(f"[INFO] collection get/create"
              f"{' → ' + collection_name if collection_name else ''}")
        return self.collection

    # ═══════════════════════════════════════════════════════════════════════
    # Storage
    # ═══════════════════════════════════════════════════════════════════════
    def m_store_fio_entry(
        self,
        fio_entries: List[Dict[str, Any]],
        batch_size: int = 64,
        skip_existing: bool = True,
        collection_name: Optional[str] = None,
    ):
        """Batch store validated FIO entries."""
        if not fio_entries:
            print("[INFO] No FIO entries to store.")
            return

        collection = self._get_collection(collection_name)

        # Validate + normalize
        cleaned_entries = []
        for fio in fio_entries:
            cleaned = validate_fio(fio)
            cleaned["document"] = build_document_string(cleaned)
            cleaned_entries.append(cleaned)
        fio_entries = cleaned_entries
        print(f"[INFO] Validated {len(fio_entries)} FIO entries.")

        # Duplicate filtering
        if skip_existing:
            try:
                existing = collection.get(include=[])
                existing_ids = set(existing.get("ids", []))
            except Exception:
                existing_ids = set()

            before = len(fio_entries)
            fio_entries = [f for f in fio_entries if f["id"] not in existing_ids]
            skipped = before - len(fio_entries)
            print(f"[INFO] Duplicate filtering | Skipped={skipped} | "
                  f"Remaining={len(fio_entries)}")
            if not fio_entries:
                print("[INFO] Nothing new to insert.")
                return

        # Batch insert
        docs_batch, metas_batch, ids_batch = [], [], []
        total_inserted = 0
        for fio in fio_entries:
            document = fio["document"]
            metadata = {
                field: fio.get(field)
                for field in CHROMADB_METADATA_FIELDS
                if field != "document"
            }
            docs_batch.append(document)
            metas_batch.append(metadata)
            ids_batch.append(fio["id"])

            if len(docs_batch) >= batch_size:
                collection.upsert(documents=docs_batch, metadatas=metas_batch, ids=ids_batch)
                total_inserted += len(docs_batch)
                print(f"[INFO] Inserted batch | total={total_inserted}")
                docs_batch.clear(); metas_batch.clear(); ids_batch.clear()

        if docs_batch:
            collection.upsert(documents=docs_batch, metadatas=metas_batch, ids=ids_batch)
            total_inserted += len(docs_batch)

        print(f"[SUCCESS] Inserted {total_inserted} FIO entries into ChromaDB")

    # ═══════════════════════════════════════════════════════════════════════
    # Semantic search  (hardened: close-reconnect-retry once, then skip)
    # ═══════════════════════════════════════════════════════════════════════
    def semantic_search(
        self,
        query: Dict[str, Any],
        k: int = 5,
        candidate_pool: int = 30,
        weights: Optional[Dict[str, float]] = None,
        collection_name: Optional[str] = None,
        where: Optional[Dict] = None,
    ) -> Dict:
        """
        Semantic retrieval with temporal filtering, reranking, and
        alpha / recency / importance support.

        On a DB read error at get_collection / count / query, the poisoned
        connection is closed and reconnected, and the read is retried ONCE.
        If it still fails, an empty result is returned so the QA loop
        continues instead of crashing.
        """
        EMPTY = {
            "ids":       [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "scores":    [[]],
        }

        weights = weights or {
            "semantic":   0.55,
            "recency":    0.20,
            "importance": 0.20,
            "alpha":      0.05,
        }

        query_text = self._build_query_text(query)
        if not query_text.strip():
            print("[WARN] EMPTY QUERY TEXT")

        # Resilient read: get_collection -> count -> query, reconnect+retry once.
        RETRIES = 1
        raw_results = None
        for attempt in range(RETRIES + 1):
            try:
                collection = self._get_collection(collection_name)

                collection_count = collection.count()
                print("COUNT OK:", collection_count)
                print(f"[INFO] semantic_search → collection={collection_name}"
                      f" | total_docs={collection_count}")

                if collection_count == 0:
                    print("[WARN] Empty collection. Returning no results.")
                    return EMPTY

                requested_results = max(k, candidate_pool)
                n_results = max(1, min(requested_results, collection_count))

                print(f"[INFO] query settings | requested={requested_results}"
                      f" | final_n_results={n_results} | k={k}"
                      f" | candidate_pool={candidate_pool}")

                raw_results = collection.query(
                    query_texts=[str(query_text)],
                    n_results=int(n_results),
                    include=["documents", "metadatas", "distances"],
                    where=where,
                )
                break  # success

            except Exception as e:
                print(f"[WARN] DB read failed (attempt {attempt}) on "
                      f"'{collection_name}': {type(e).__name__}: {e}")
                print(f"       where={where} | query_text={query_text!r}")
                if attempt < RETRIES:
                    print("[INFO] reconnecting client and retrying once...")
                    self._reconnect()
                else:
                    print("[WARN] retry exhausted — skipping this query, "
                          "returning no results.")
                    return EMPTY

        # Extract results safely
        ids       = raw_results.get("ids",       [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        if not ids:
            print("[INFO] No matching results found.")
            return EMPTY

        print(f"[INFO] Retrieved {len(ids)} candidates")

        return self._rerank_results(
            query=query,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            distances=distances,
            k=min(k, len(ids)),
            weights=weights,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Debug helpers
    # ═══════════════════════════════════════════════════════════════════════
    def show_entries(self, n: int = 5, collection_name: Optional[str] = None):
        collection = self._get_collection(collection_name)
        data = collection.get(include=["documents", "metadatas"])
        total = len(data.get("ids", []))

        print(f"\n[ChromaFIOStorage] Total entries: {total}")
        print("-" * 80)
        for i in range(min(n, total)):
            print(f"Entry {i+1}")
            print(f"ID:        {data['ids'][i]}")
            print(f"Document:  {data['documents'][i]}")
            print(f"Metadata:  {data['metadatas'][i]}")
            print("-" * 80) 
#updated

if __name__ == "__main__":
    storage = ChromaFIOStorage(db_dir="./BUILD_7day_ablation_full")
    storage.show_entries(n=3)