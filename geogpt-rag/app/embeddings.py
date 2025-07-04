import torch, os
from pydantic import BaseModel, Field
from typing import List, Any
from langchain.embeddings.base import Embeddings

from .models.embedding import EmbeddingModel
from .config import EMBED_MODEL, EMBEDDING_BATCH_SIZE


class GeoEmbeddings(BaseModel, Embeddings):
    """Langchain-compatible embeddings wrapper using in-process EmbeddingModel.
    
    Preserves the original batching logic while using environment configuration.
    """
    
    # Declare the embedding_model field for Pydantic validation
    embedding_model: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        device = os.getenv("EMBEDDING_DEVICE", "cuda")

        fp16 = (
            os.getenv("EMBEDDING_FP16", "false").lower() == "true" and device != "cpu"
        )

        self.embedding_model = EmbeddingModel(
            model_name=EMBED_MODEL,
            device=device,
            fp16=fp16
        )

    def batch_embedding(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding for passages - replaces the HTTP call logic."""
        try:
            # Use encode_passages for document/passage embedding
            embeddings = self.embedding_model.encode_passages(
                texts, 
                batch_size=EMBEDDING_BATCH_SIZE
            )
            return embeddings
        except Exception as e:
            raise ConnectionError(f"Embedding model error: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with the same batching logic as the original."""
        # Clean texts by replacing newlines with spaces (same as original)
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        
        # Use original batching logic
        if len(texts) <= EMBEDDING_BATCH_SIZE:
            return self.batch_embedding(texts)
        
        embeddings = []
        for i in range(int(len(texts) / EMBEDDING_BATCH_SIZE) + int(len(texts) % EMBEDDING_BATCH_SIZE > 0)):
            end_index = min((i + 1) * EMBEDDING_BATCH_SIZE, len(texts))
            batch_embeddings = self.batch_embedding(texts[i * EMBEDDING_BATCH_SIZE: end_index])
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query - replaces the HTTP call logic."""
        # Clean text (same as original)
        text = text.replace("\n", " ")
        
        try:
            # Use encode_queries for query embedding
            embeddings = self.embedding_model.encode_queries(
                [text], 
                batch_size=EMBEDDING_BATCH_SIZE
            )
            
            if not embeddings:
                raise ValueError("q_embeddings not in response")
            return embeddings[0]
        except Exception as e:
            raise ConnectionError(f"Embedding model error: {e}")

    class Config:
        arbitrary_types_allowed = True  # Allow EmbeddingModel instance