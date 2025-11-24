"""Embedder abstraction using SentenceTransformer.

Usage:
    from embedder import Embedder
    e = Embedder(model_name='all-MiniLM-L6-v2')
    vectors = e.encode(['text1', 'text2'], batch_size=32)
"""
from typing import List, Iterable
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    # graceful fallback for editor linting; runtime requires package installed
    SentenceTransformer = None


class Embedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        if SentenceTransformer is None:
            raise RuntimeError('sentence-transformers is required. Install from requirements.txt')
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        """Encode an iterable of texts into a numpy array of float32 vectors.

        Returns an array shape (N, D) dtype float32.
        """
        # sentence-transformers returns numpy arrays when convert_to_numpy=True
        vectors = self.model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype='float32')
