"""
Configuration module: replicates GeoGPT-RAG knobs but lets you override
everything via environment variables.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Project paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

# Create upload directory with error handling
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    # If we can't create the directory due to permissions, 
    # it will be created by the application when needed
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Embedding / Reranking
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))
RERANKING_BATCH_SIZE: int = int(os.getenv("RERANKING_BATCH_SIZE", 32))

# Keep old constant name so existing code works
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "GeoGPT-Research-Project/GeoEmbedding")
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "GeoGPT-Research-Project/GeoReranker")

# ─────────────────────────────────────────────────────────────────────────────
# Sentence-piece / chunker
# ─────────────────────────────────────────────────────────────────────────────
BERT_PATH: str = os.getenv("BERT_PATH", "bert-base-uncased")
MAX_SIZE: int = int(os.getenv("CHUNK_MAX_SIZE", 512))

# ─────────────────────────────────────────────────────────────────────────────
# Milvus / Zilliz Cloud
# ─────────────────────────────────────────────────────────────────────────────
ZILLIZ_URI: str | None = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN: str | None = os.getenv("ZILLIZ_TOKEN")
MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "geodocs")
CONNECTION_ARGS: dict[str, str | None] = {"uri": ZILLIZ_URI, "token": ZILLIZ_TOKEN}

# ─────────────────────────────────────────────────────────────────────────────
# Reranker / Retrieval heuristics
# ─────────────────────────────────────────────────────────────────────────────
VEC_RECALL_NUM: int = int(os.getenv("VEC_RECALL_NUM", 128))
TOP_K: int = int(os.getenv("TOP_K", 3))
META: bool = os.getenv("META", "true").lower() == "true"
SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", 1.5))

# ─────────────────────────────────────────────────────────────────────────────
# Chunk-expansion
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_PATH_NAME: str = os.getenv("CHUNK_PATH_NAME", "split_chunks")
EXPAND_RANGE: int = int(os.getenv("EXPAND_RANGE", 1024))
EXPAND_TIME_OUT: int = int(os.getenv("EXPAND_TIME_OUT", 30))

# ─────────────────────────────────────────────────────────────────────────────
# External LLM Configuration
# ─────────────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai-compatible")  # openai-compatible | sagemaker

# OpenAI-Compatible Configuration
LLM_URL: str = os.getenv("LLM_URL", "")
LLM_KEY: str = os.getenv("LLM_KEY", "")

# SageMaker Configuration
SAGEMAKER_ENDPOINT_NAME: str = os.getenv("SAGEMAKER_ENDPOINT_NAME", "")
AWS_REGION: str = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
AWS_ACCESS_KEY_ID: str | None = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY: str | None = os.getenv("AWS_SECRET_ACCESS_KEY")

# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek-style RAG prompt
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_RAG_PROMPT: str = """\
# The following contents are the search results related to the user's message:
{search_results}
In the search results I provide to you, each result is formatted as [document X begin]...[document X end], where X represents the numerical index of each article. Please cite the context at the end of the relevant sentence when appropriate. Use the citation format [citation:X] in the corresponding part of your answer. If a sentence is derived from multiple contexts, list all relevant citation numbers, such as [citation:3][citation:5]. Be sure not to cluster all citations at the end; instead, include them in the corresponding parts of the answer.
When responding, please keep the following points in mind:
- Today is {cur_date}.
- Not all content in the search results is closely related to the user's question. You need to evaluate and filter the search results based on the question.
- If all the search results are irrelevant, please answer the question by yourself professionally and concisely.
- The search results may focus only on a few points; use the information provided, but do not favour those points in your answer. Reason comprehensively.
- For listing-type questions (e.g., listing all flight information), try to limit the answer to 10 key points and inform the user that they can refer to the search sources for complete information. Prioritise the most complete and relevant items. Avoid mentioning content not provided in the search results unless necessary.
- For creative tasks (e.g., writing an essay), ensure references are cited within the text, e.g. [citation:3][citation:5], rather than only at the end. Extract key information and generate an insightful, creative, professional answer. Extend length as needed, covering each point in detail from multiple perspectives.
- If the response is lengthy, structure and summarise it in paragraphs. If a point-by-point format is needed, limit to ~5 points and merge related content.
- For objective Q&A, if the answer is very brief, you may add one or two related sentences to enrich it.
- Use a suitable, visually appealing format for readability.
- Synthesize information from multiple relevant documents; avoid repetitive citations.
- Unless the user requests otherwise, reply in the user's language.
# The user's message is:
{question}
"""

RAG_PROMPT: str = os.getenv("RAG_PROMPT", DEFAULT_RAG_PROMPT)

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def today() -> str:
    """Return today's date in ISO format (YYYY-MM-DD)."""
    return date.today().isoformat()
