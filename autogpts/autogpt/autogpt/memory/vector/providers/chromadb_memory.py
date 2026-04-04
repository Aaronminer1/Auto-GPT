"""ChromaDB-backed vector memory provider.

Stores memories in a local persistent ChromaDB database (SQLite + HNSW).
Embeddings are generated with the local sentence-transformers model — no API key needed.
Data is persisted under:  <workspace>/chroma_memory/
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from autogpt.config import Config

from ..memory_item import MemoryItem, MemoryItemRelevance
from ..utils import Embedding, get_embedding
from .base import VectorMemoryProvider

logger = logging.getLogger(__name__)

COLLECTION_NAME = "autogpt_memory"


class ChromaMemory(VectorMemoryProvider):
    """Persistent vector memory backed by ChromaDB with local embeddings."""

    def __init__(self, config: Config) -> None:
        import chromadb
        from chromadb.config import Settings

        persist_dir = str(config.workspace_path / "chroma_memory")
        logger.debug(f"ChromaMemory: persisting to {persist_dir}")

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        # In-memory cache of MemoryItems (keyed by chroma id) for __iter__ /
        # scoring support.  Rebuilt lazily from the stored raw_content on load.
        self._items: dict[str, MemoryItem] = {}
        self._config = config
        self._reload_items()

    # ------------------------------------------------------------------
    # MutableSet interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[MemoryItem]:
        return iter(self._items.values())

    def __contains__(self, x: object) -> bool:
        if not isinstance(x, MemoryItem):
            return False
        return any(
            i.raw_content == x.raw_content for i in self._items.values()
        )

    def __len__(self) -> int:
        return self._collection.count()

    def add(self, item: MemoryItem) -> int:
        item_id = str(uuid.uuid4())
        embedding = item.e_summary if hasattr(item, "e_summary") else get_embedding(
            item.summary, self._config
        )
        # Normalise to plain list[float]
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        self._collection.add(
            ids=[item_id],
            embeddings=[embedding],
            documents=[item.raw_content],
            metadatas=[{
                "summary": item.summary[:500],
                "source_type": item.metadata.get("source_type", ""),
            }],
        )
        self._items[item_id] = item
        logger.debug(f"ChromaMemory: added item {item_id} ({item.metadata.get('source_type', '')})")
        return len(self._items)

    def discard(self, item: MemoryItem) -> None:
        for item_id, stored in list(self._items.items()):
            if stored.raw_content == item.raw_content:
                self._collection.delete(ids=[item_id])
                del self._items[item_id]
                break

    def clear(self) -> None:
        """Delete all memories from the collection."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._items.clear()
        logger.debug("ChromaMemory: cleared all memories")

    # ------------------------------------------------------------------
    # Retrieval — delegate to ChromaDB's HNSW index for speed
    # ------------------------------------------------------------------

    def get_relevant(
        self, query: str, k: int, config: Config
    ) -> Sequence[MemoryItemRelevance]:
        if len(self) < 1:
            return []

        q_embedding = get_embedding(query, config)
        if isinstance(q_embedding, np.ndarray):
            q_embedding = q_embedding.tolist()

        n = min(k, len(self))
        results = self._collection.query(
            query_embeddings=[q_embedding],
            n_results=n,
            include=["documents", "distances", "embeddings"],
        )

        relevances = []
        ids = results["ids"][0]
        distances = results["distances"][0]

        for item_id, dist in zip(ids, distances):
            # ChromaDB cosine distance → similarity score (1 - distance)
            score = float(1.0 - dist)
            item = self._items.get(item_id)
            if item is None:
                continue
            relevances.append(
                MemoryItemRelevance(
                    memory_item=item,
                    for_query=query,
                    summary_relevance_score=score,
                    chunk_relevance_scores=[score],
                )
            )

        return relevances

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reload_items(self) -> None:
        """Reload in-memory item cache from the persisted collection."""
        try:
            results = self._collection.get(include=["documents", "embeddings", "metadatas"])
        except Exception as e:
            logger.warning(f"ChromaMemory: could not reload items: {e}")
            return

        ids = results.get("ids") or []
        docs = results.get("documents") or []
        _emb = results.get("embeddings")
        embeddings = [] if (_emb is None or len(_emb) == 0) else _emb
        metadatas = results.get("metadatas") or []

        for i, item_id in enumerate(ids):
            try:
                raw = docs[i] if i < len(docs) else ""
                emb = embeddings[i] if i < len(embeddings) else []
                meta = metadatas[i] if i < len(metadatas) else {}
                summary = meta.get("summary", raw[:200])

                self._items[item_id] = MemoryItem(
                    raw_content=raw,
                    summary=summary,
                    chunks=[raw],
                    chunk_summaries=[summary],
                    e_summary=emb,
                    e_chunks=[emb],
                    metadata=dict(meta),
                )
            except Exception as e:
                logger.debug(f"ChromaMemory: skipping item {item_id}: {e}")

        logger.debug(f"ChromaMemory: loaded {len(self._items)} items from disk")
