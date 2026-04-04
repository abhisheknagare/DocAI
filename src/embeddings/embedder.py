import pickle
import numpy as np
from pathlib import Path
from typing import Union, Optional


class Embedder:

    def __init__(self, model_path: str = "data/index/tfidf.pkl",
                 prefer_transformers: bool = False):
        self._tfidf = None
        self._svd = None
        self._dim = 128
        self._st_model = None
        self._model_path = model_path

        if prefer_transformers:
            self._try_load_transformers()

        if self._st_model is None:
            self._load_tfidf(model_path)

    def _try_load_transformers(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._dim = 384
            print("  Using sentence-transformers (all-MiniLM-L6-v2)")
        except Exception:
            pass

    def _load_tfidf(self, path: str):
        p = Path(path)
        if p.exists():
            with open(p, "rb") as f:
                obj = pickle.load(f)
            self._tfidf = obj["tfidf"]
            self._svd = obj["svd"]
            self._dim = obj["dim"]
            print(f"  Loaded TF-IDF+SVD embedder (dim={self._dim})")
        else:
            raise FileNotFoundError(
                f"Embedder model not found at {path}. "
                "Run ingestion first: python scripts/ingest.py"
            )

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, texts: list[str]):
        """No-op — model is pre-fitted."""
        pass

    def embed(self, texts: Union[str, list[str]],
              show_progress: bool = False, batch_size: int = 64) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if self._st_model is not None:
            vecs = self._st_model.encode(texts, normalize_embeddings=True,
                                          convert_to_numpy=True, show_progress_bar=False)
            return vecs.astype(np.float32)

        mat = self._tfidf.transform(texts)
        vecs = self._svd.transform(mat).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


_cache: Optional[Embedder] = None

def get_embedder(model_name: str = "minilm",
                 model_path: str = "data/index/tfidf.pkl") -> Embedder:
    global _cache
    if _cache is None:
        _cache = Embedder(model_path=model_path)
    return _cache
