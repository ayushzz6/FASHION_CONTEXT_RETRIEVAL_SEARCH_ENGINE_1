"""Vector store adapter built on ChromaDB."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:  # pragma: no cover - optional in lightweight envs
    chromadb = None
    Settings = None


class ChromaVectorStore:
    """Minimal wrapper around Chroma for global embeddings."""

    def __init__(self, persist_directory: str, collection_name: str = "fashion_global") -> None:
        if chromadb is None:
            raise ImportError("chromadb is not installed. Please `pip install chromadb`.")
        client = chromadb.PersistentClient(path=persist_directory, settings=Settings(allow_reset=True))
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadatas: Iterable[Dict[str, Any]] | None = None,
    ) -> None:
        self.collection.add(
            ids=list(ids),
            embeddings=embeddings.astype(np.float32).tolist(),
            metadatas=list(metadatas) if metadatas is not None else None,
        )

    def search(self, embedding: np.ndarray, top_n: int = 50) -> List[Dict[str, Any]]:
        """Return list of {id, distance, metadata} sorted by distance ascending (cosine)."""
        result = self.collection.query(query_embeddings=embedding.astype(np.float32).tolist(), n_results=top_n)
        ids = result.get("ids", [[]])[0]
        dists = result.get("distances", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        out: List[Dict[str, Any]] = []
        for i, d, m in zip(ids, dists, metas):
            out.append({"id": i, "distance": d, "metadata": m or {}})
        return out
