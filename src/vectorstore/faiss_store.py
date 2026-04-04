import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    filename: str
    company: str
    text: str
    section: str
    score: float
    chunk_index: int
    total_chunks: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "company": self.company,
            "text": self.text,
            "section": self.section,
            "score": self.score,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


class FAISSVectorStore:

    def __init__(self, dim: int, index_dir: str = "data/index"):
        self.dim = dim
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._index: Optional[faiss.Index] = None
        self._metadata: list[dict] = []   
        self._id_to_row: dict[str, int] = {}

        self._index_path = self.index_dir / "faiss.index"
        self._meta_path = self.index_dir / "metadata.jsonl"


    def build(self, embeddings: np.ndarray, metadata: list[dict]):
        
        assert len(embeddings) == len(metadata), "Length mismatch"
        assert embeddings.dtype == np.float32, "Need float32"

        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(embeddings)
        self._metadata = list(metadata)
        self._id_to_row = {m["chunk_id"]: i for i, m in enumerate(metadata)}

        print(f"  ✓ Built FAISS index: {self._index.ntotal} vectors, dim={self.dim}")

    def add(self, embeddings: np.ndarray, metadata: list[dict]):
        if self._index is None:
            self.build(embeddings, metadata)
            return
        start = self._index.ntotal
        self._index.add(embeddings)
        for i, m in enumerate(metadata):
            self._metadata.append(m)
            self._id_to_row[m["chunk_id"]] = start + i


    def save(self):
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "w") as f:
            for m in self._metadata:
                f.write(json.dumps(m) + "\n")
        print(f"   Saved index: {self._index.ntotal} vectors → {self.index_dir}")

    def load(self) -> bool:
        if not self._index_path.exists() or not self._meta_path.exists():
            return False
        self._index = faiss.read_index(str(self._index_path))
        self._metadata = []
        with open(self._meta_path) as f:
            for line in f:
                self._metadata.append(json.loads(line))
        self._id_to_row = {m["chunk_id"]: i for i, m in enumerate(self._metadata)}
        print(f"   Loaded index: {self._index.ntotal} vectors from {self.index_dir}")
        return True

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def num_vectors(self) -> int:
        return 0 if self._index is None else self._index.ntotal

    @property
    def documents(self) -> list[str]:
        """List of unique document IDs in the index."""
        return list({m["doc_id"] for m in self._metadata})


    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        filter_doc_id: Optional[str] = None,
    ) -> list[SearchResult]:
        
        if not self.is_ready:
            raise RuntimeError("Vector store not initialized. Run ingestion first.")

        q = query_vec.reshape(1, -1).astype(np.float32)
        fetch_k = top_k * 5 if filter_doc_id else top_k

        scores, indices = self._index.search(q, min(fetch_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            if filter_doc_id and meta["doc_id"] != filter_doc_id:
                continue
            results.append(
                SearchResult(
                    chunk_id=meta["chunk_id"],
                    doc_id=meta["doc_id"],
                    filename=meta["filename"],
                    company=meta["company"],
                    text=meta["text"],
                    section=meta.get("section", "General"),
                    score=float(score),
                    chunk_index=meta.get("chunk_index", 0),
                    total_chunks=meta.get("total_chunks", 0),
                )
            )
            if len(results) >= top_k:
                break

        return results

    def get_document_chunks(self, doc_id: str, max_chunks: int = 50) -> list[dict]:
        chunks = [
            m for m in self._metadata
            if m["doc_id"] == doc_id
        ]
        chunks.sort(key=lambda x: x.get("chunk_index", 0))
        return chunks[:max_chunks]

    def stats(self) -> dict:
        return {
            "total_vectors": self.num_vectors,
            "documents": len(self.documents),
            "dim": self.dim,
            "document_list": self.documents,
        }
