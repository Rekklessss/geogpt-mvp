import os
from pydantic import BaseModel, Field
from typing import List, Tuple, Any

from .models.reranker import RerankerModel
from .config import RERANK_MODEL, RERANKING_BATCH_SIZE


class GeoReRanking(BaseModel):
    """Reranking wrapper using in-process RerankerModel.
    
    Preserves the original batching logic while using environment configuration.
    """
    
    # Declare the reranker_model field for Pydantic validation
    reranker_model: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Get device and fp16 settings from environment or use defaults
        device = os.getenv("RERANKING_DEVICE", "cpu")
        fp16 = os.getenv("RERANKING_FP16", "true").lower() == "true"
        
        self.reranker_model = RerankerModel(
            model_name=RERANK_MODEL,
            device=device,
            fp16=fp16
        )

    def batch_reranking(self, qp_pairs: List[List[str]]) -> List[float]:
        """Batch reranking for query-passage pairs - replaces the HTTP call logic."""
        try:
            # Convert List[List[str]] to List[Tuple[str, str]] as expected by RerankerModel
            tuple_pairs = [(pair[0], pair[1]) for pair in qp_pairs]
            
            scores = self.reranker_model.compute_scores(tuple_pairs, normalize=True)
            
            # Ensure we return a list even for single scores
            if isinstance(scores, float):
                scores = [scores]
            return scores
        except Exception as e:
            raise ValueError(f"Reranking model error: {e}")

    def compute_scores(self, qp_pairs: List[List[str]]) -> List[float]:
        """Compute reranking scores with the same batching logic as the original."""
        # Use original batching logic
        if len(qp_pairs) <= RERANKING_BATCH_SIZE:
            return self.batch_reranking(qp_pairs)
        
        scores = []
        for i in range(int(len(qp_pairs) / RERANKING_BATCH_SIZE) + int(len(qp_pairs) % RERANKING_BATCH_SIZE > 0)):
            end_index = min((i + 1) * RERANKING_BATCH_SIZE, len(qp_pairs))
            batch_scores = self.batch_reranking(qp_pairs[i * RERANKING_BATCH_SIZE: end_index])
            scores.extend(batch_scores)
        return scores

    class Config:
        arbitrary_types_allowed = True  # Allow RerankerModel instance
