import torch
from typing import List, Optional

DEFAULT_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class EmbeddingModel:
    """Wrapper around Sentence‑Transformers that mimics the GEOembedding API.

    The public methods mirror the original /query and /passage endpoints so we can
    keep exactly the same prompting logic while running *in‑process* instead of
    over HTTP.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        task_description: str = DEFAULT_TASK,
        fp16: bool = False,
    ) -> None:
        # Lazy import to avoid issues during module loading
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(f"sentence_transformers is required but not installed: {e}")
            
        model_kwargs = {"torch_dtype": torch.float16} if fp16 else {}
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
            model_kwargs=model_kwargs or None,
        )
        self.task = task_description

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _wrap_instruction(task: str, text: str) -> str:
        """Replicates the original get_detailed_instruct() helper."""
        return f"Instruct: {task}\nQuery: {text}"

    def _encode(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        processed = (
            [self._wrap_instruction(instruction or self.task, t) for t in texts]
            if instruction or self.task
            else texts
        )
        return self.model.encode(
            processed,
            convert_to_tensor=False,
            normalize_embeddings=True,
            batch_size=batch_size,
        ).tolist()

    # ------------------------------------------------------------------
    # Public API (mirrors HTTP micro‑service)
    # ------------------------------------------------------------------
    def encode_queries(
        self,
        queries: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        return self._encode(queries, instruction, batch_size)

    def encode_passages(
        self,
        passages: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        # Passages only get wrapped if instruction is explicitly provided
        if instruction:
            processed = [self._wrap_instruction(instruction, p) for p in passages]
        else:
            processed = passages  # No wrapping for passages without instruction
        
        return self.model.encode(
            processed,
            convert_to_tensor=False,
            normalize_embeddings=True,
            batch_size=batch_size,
        ).tolist()

    # Backwards‑compat: old code calls .encode() generically
    def encode(self, texts, **kwargs):  # type: ignore[override]
        return self.encode_queries(texts, **kwargs)