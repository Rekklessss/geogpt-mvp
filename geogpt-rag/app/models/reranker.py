from __future__ import annotations

from typing import List, Tuple


class RerankerModel:
    """Thin wrapper around FlagEmbedding's FlagReranker so we can use it
    in‑process (no FastAPI micro‑service) but still keep the original API.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        fp16: bool = True,
    ) -> None:
        # Lazy import to avoid issues during module loading
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as e:
            raise ImportError(f"FlagEmbedding is required but not installed: {e}")
            
        # FlagEmbedding auto‑detects CUDA; fp16=True only matters on GPU
        self.model = FlagReranker(model_name, use_fp16=fp16)

    # ------------------------------------------------------------------
    # Public API – mirrors /query endpoint from reranker_fast_api.py
    # ------------------------------------------------------------------
    def compute_scores(
        self,
        qp_pairs: List[Tuple[str, str]],
        normalize: bool = True,
    ) -> List[float]:
        """Score each (query, passage) pair.

        Args:
            qp_pairs: list of 2‑tuples (q, p)
            normalize: if True reproduce original `normalize=True` behaviour
        Returns:
            list of floats – higher is more relevant (cosine‑like score)
        """
        return self.model.compute_score(qp_pairs, normalize=normalize)

    # Back‑compat alias used in the original API response key
    def rerank(self, qp_pairs: List[Tuple[str, str]], normalize: bool = True):
        return self.compute_scores(qp_pairs, normalize)