"""Sanity-check a built Chroma FIO store.

``analyze_chroma_dates(path)`` opens the persistent Chroma DB at ``path`` and,
for every collection, prints the entry count and the number of entries per
``date`` metadata value (DAY1 ... DAY7). Called at the end of
scripts/build_memory.py on the raw and opt stores.
"""

import chromadb
from collections import Counter


def analyze_chroma_dates(collection_path):
    """
    Analyze date metadata distribution across all ChromaDB collections.

    Args:
        collection_path (str): Path to ChromaDB persistent storage.
    """

    client = chromadb.PersistentClient(path=collection_path)

    collections = client.list_collections()
    #print(f"Total collections: {len(collections)}\n")

    for col in collections:

        collection = client.get_collection(col.name)
        count = collection.count()

        # skip empty collections
        if count == 0:
            print(f"\nCollection: {col.name} is empty. Skipping.")
            continue

        print(f"\nCollection: {col.name}")
        print(f"Total Entries: {count}")

        # fetch metadata
        data = collection.get(include=["metadatas"])

        metadatas = data.get("metadatas", [])

        # collection-wise date count
        date_counter = Counter()

        for md in metadatas:

            if md is None:
                continue

            date_value = md.get("date")

            if date_value is not None:
                date_counter[str(date_value)] += 1
        print("\nDay-wise Entry Count:")

        for day, freq in sorted(date_counter.items()):
            print(f"  {day}: {freq}")
        print(f"Total entries with date metadata: {sum(date_counter.values())}") # total entries with date metadata

        print("-" * 50)



# =========================
# Example Usage
# =========================
if __name__ == "__main__":

    analyze_chroma_dates("./memory_build_Jake_7daysFinal/chroma_fio_opt")