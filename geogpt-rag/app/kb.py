"""
Knowledge Base Document Question Answering system.

This module provides the main KBDocQA class that handles document ingestion,
vector storage, retrieval, reranking, and LLM-based question answering.
"""

from __future__ import annotations

import json
import os
import time
import logging
from datetime import date
from multiprocessing import Process, Queue
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import openai
from langchain.vectorstores import Milvus
from pymilvus import Collection

from .embeddings import GeoEmbeddings
from .reranking import GeoReRanking
from .utils.parsers import split_text
from .models.sagemaker_llm import SageMakerLLMClient
from .config import (
    CONNECTION_ARGS, 
    MILVUS_COLLECTION, 
    CHUNK_PATH_NAME, 
    MAX_SIZE, 
    VEC_RECALL_NUM,
    EXPAND_TIME_OUT, 
    META, 
    EXPAND_RANGE, 
    TOP_K, 
    SCORE_THRESHOLD, 
    LLM_PROVIDER,
    LLM_URL, 
    LLM_KEY, 
    SAGEMAKER_ENDPOINT_NAME,
    AWS_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    RAG_PROMPT,
    BASE_DIR,
    today
)


# Setup logging
def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "rag.log")
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()  # Also log to console
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# Ensure chunk directory exists
chunk_dir = Path(CHUNK_PATH_NAME)
chunk_dir.mkdir(parents=True, exist_ok=True)


def generate_metadata(doc: Dict[str, Any]) -> str:
    """Generate metadata string from document fields."""
    meta_list = []
    for field in ['title', 'section', 'subsection']:
        if doc.get(field, ""):
            meta_list.append(doc[field])
    return ",".join(meta_list)


def filter_documents_by_score(docs: List[Dict[str, Any]], score_threshold: float) -> List[Dict[str, Any]]:
    """Filter documents based on score threshold."""
    return [doc for doc in docs if doc.get("score", float('inf')) >= score_threshold]


def generate_llm_response(prompt: str) -> str:
    """Generate response using external LLM (OpenAI-compatible or SageMaker)."""
    
    if LLM_PROVIDER.lower() == "sagemaker":
        return _generate_sagemaker_response(prompt)
    else:
        return _generate_openai_compatible_response(prompt)


def _generate_openai_compatible_response(prompt: str) -> str:
    """Generate response using OpenAI-compatible API."""
    if not LLM_URL or not LLM_KEY:
        logger.warning("LLM_URL or LLM_KEY not configured, returning fallback response")
        return "I apologize, but I cannot generate a response as the LLM service is not properly configured."
    
    try:
        client = openai.Client(base_url=LLM_URL, api_key=LLM_KEY)
        models = client.models.list()
        
        if not models.data:
            logger.error("No models available from LLM service")
            return "I apologize, but no language models are available at the moment."
        
        model = models.data[0].id
        messages = [{"role": "user", "content": prompt}]
        
        # Retry mechanism for connection issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    top_p=0.8,
                    max_tokens=32768,
                    timeout=3600
                )
                return response.choices[0].message.content
            
            except Exception as e:
                logger.error(f"OpenAI-compatible LLM generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise
                    
    except Exception as e:
        logger.error(f"OpenAI-compatible LLM generation error: {e}")
        return f"I apologize, but I encountered an error while generating a response: {str(e)}"


def _generate_sagemaker_response(prompt: str) -> str:
    """Generate response using AWS SageMaker endpoint."""
    if not SAGEMAKER_ENDPOINT_NAME:
        logger.warning("SAGEMAKER_ENDPOINT_NAME not configured, returning fallback response")
        return "I apologize, but I cannot generate a response as the SageMaker endpoint is not properly configured."
    
    try:
        # Initialize SageMaker client
        sagemaker_client = SageMakerLLMClient(
            endpoint_name=SAGEMAKER_ENDPOINT_NAME,
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # Generate response
        response = sagemaker_client.generate(
            prompt=prompt,
            max_tokens=32768,
            temperature=0.7,
            top_p=0.8,
            timeout=300
        )
        
        return response
        
    except Exception as e:
        logger.error(f"SageMaker LLM generation error: {e}")
        return f"I apologize, but I encountered an error while generating a response: {str(e)}"


class KBDocQA:
    """
    Knowledge Base Document Question Answering system.
    
    Handles document ingestion, vector storage, retrieval, reranking,
    and LLM-based question answering with context expansion.
    """
    
    def __init__(
        self,
        connection_args: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> None:
        """
        Initialize the KB document QA system.
        
        Args:
            connection_args: Milvus connection arguments
            collection_name: Name of the Milvus collection
        """
        self.embeddings = GeoEmbeddings()
        self.reranking = GeoReRanking()
        
        self.milvus_connection_args = connection_args or CONNECTION_ARGS
        self.collection_name = collection_name or MILVUS_COLLECTION
        
        # Initialize vector store
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args=self.milvus_connection_args,
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64},
            },
            drop_old=False,
            auto_id=True
        )
        
        logger.info(f"Initialized KBDocQA with collection: {self.collection_name}")

    def add_file(self, file_path: str, max_size: int = MAX_SIZE) -> None:
        """
        Add a file to the knowledge base.
        
        Args:
            file_path: Path to the file to add
            max_size: Maximum chunk size in tokens
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Adding file to KB: {file_path}")
        
        # Read and split text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunk_data = split_text(text, str(file_path), max_size=max_size)
        
        # Save chunk data
        filename = file_path.stem
        chunk_file_path = Path(CHUNK_PATH_NAME) / f"{filename}.jsonl"
        
        with open(chunk_file_path, 'w', encoding='utf-8') as f:
            for doc in chunk_data:
                doc["chunk_path"] = str(chunk_file_path)
                f.write(json.dumps(doc, ensure_ascii=False))
                f.write("\n")
        
        # Prepare data for vector store
        texts = []
        metadata = []
        
        for doc in chunk_data:
            texts.append(doc["text"])
            # Remove text from metadata to avoid duplication
            metadata_doc = {k: v for k, v in doc.items() if k != "text"}
            metadata.append(metadata_doc)
        
        # Add to vector store
        self.vector_store.add_texts(texts, metadatas=metadata)
        
        logger.info(f"Successfully added {len(texts)} chunks from {file_path}")

    def drop_collection(self) -> None:
        """Drop the current Milvus collection."""
        if isinstance(self.vector_store.col, Collection):
            self.vector_store.col.drop()
            self.vector_store.col = None
            logger.info(f"Dropped collection: {self.collection_name}")

    def vector_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of document dictionaries with scores
        """
        start_time = time.time()
        
        chunks = self.vector_store.similarity_search_with_score(
            query, 
            k=k,
            param={"metric_type": "COSINE", "params": {"ef": 4096}}
        )
        
        results = []
        for doc, score in chunks:
            doc_info = doc.metadata.copy()
            doc_info["emb_dist"] = score
            doc_info["text"] = doc.page_content
            results.append(doc_info)
        
        search_time = time.time() - start_time
        logger.info(f'Vector search time: {search_time:.3f}s')
        
        return results

    def retrieval(
        self, 
        query: str, 
        k: int = TOP_K, 
        expand_len: int = EXPAND_RANGE, 
        score_threshold: float = SCORE_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Complete retrieval pipeline with reranking and expansion.
        
        Args:
            query: Search query
            k: Number of final results
            expand_len: Length for context expansion
            score_threshold: Minimum score threshold
            
        Returns:
            List of retrieved and processed documents
        """
        # Initial vector search
        initial_chunks = self.vector_search(query, k=VEC_RECALL_NUM)
        
        if not initial_chunks:
            logger.warning("No documents found in vector search")
            return []
        
        # Rerank documents
        top_docs = self.rerank_documents(query, initial_chunks, k, score_threshold)
        
        # Expand context
        results = self.expand_documents(top_docs, expand_len)
        
        return results

    def rerank_documents(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        k: int, 
        score_threshold: float
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank documents using the reranking model.
        
        Args:
            query: Original query
            chunks: List of candidate documents
            k: Number of top documents to return
            score_threshold: Minimum score threshold
            
        Returns:
            List of (document, score) tuples
        """
        start_time = time.time()
        
        if not chunks:
            return []

        # Prepare query-passage pairs
        if META:
            qp_pairs = []
            for doc in chunks:
                metadata_text = generate_metadata(doc)
                full_text = f"{metadata_text}\n{doc['text']}" if metadata_text else doc['text']
                qp_pairs.append([query, full_text])
        else:
            qp_pairs = [[query, doc['text']] for doc in chunks]
        
        # Get reranking scores
        scores = self.reranking.compute_scores(qp_pairs)
        
        # Sort by score (descending)
        ranked_docs = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        
        rerank_time = time.time() - start_time
        logger.info(f'Reranking time: {rerank_time:.3f}s')
        
        # Filter by score threshold and return top k
        filtered_docs = [(doc, score) for doc, score in ranked_docs if score >= score_threshold]
        return filtered_docs[:k]

    def expand_documents(
        self, 
        docs: List[Tuple[Dict[str, Any], float]], 
        expand_len: int
    ) -> List[Dict[str, Any]]:
        """
        Expand document context using multiprocessing.
        
        Args:
            docs: List of (document, score) tuples
            expand_len: Maximum length for expansion
            
        Returns:
            List of expanded documents
        """
        start_time = time.time()
        
        if expand_len <= 0:
            # No expansion needed
            results = []
            for doc, score in docs:
                doc["score"] = score
                results.append(doc)
            return results
        
        # Parallel expansion using multiprocessing
        results = []
        jobs = []
        queues = []
        
        for doc, score in docs:
            queue = Queue()
            process = Process(
                target=self.expand_document, 
                args=(doc, expand_len, score, queue)
            )
            jobs.append(process)
            queues.append(queue)
            process.start()
        
        # Collect results with timeout
        for i, queue in enumerate(queues):
            try:
                expanded_doc = queue.get(timeout=EXPAND_TIME_OUT)
                results.append(expanded_doc)
            except Exception as e:
                logger.error(f'Document expansion timeout/error: {e}')
                # Fallback to original document
                doc, score = docs[i]
                doc["score"] = score
                results.append(doc)
                
                # Clean up process
                jobs[i].terminate()
                time.sleep(0.1)
                if not jobs[i].is_alive():
                    jobs[i].join(timeout=1.0)
            finally:
                queue.close()
        
        expand_time = time.time() - start_time
        logger.info(f'Document expansion time: {expand_time:.3f}s')
        
        return results

    def _expand_chunk_range(
        self, 
        chunks: Dict[int, Dict[str, Any]], 
        doc: Dict[str, Any], 
        expand_range: List[int], 
        expand_len: int
    ) -> Tuple[set, bool]:
        """Helper method to expand chunk range."""
        available_indices = list(chunks.keys())
        current_length = doc["length"]
        included_indices = {int(doc["index"])}
        
        for idx in expand_range:
            if idx not in chunks:
                continue
            if min(available_indices) <= idx <= max(available_indices):
                if chunks[idx]["length"] + current_length > expand_len:
                    return included_indices, True
                else:
                    current_length += chunks[idx]["length"]
                    included_indices.add(idx)
        
        return included_indices, False

    def _find_expansion_indices(
        self, 
        chunks: Dict[int, Dict[str, Any]], 
        doc: Dict[str, Any], 
        expand_len: int
    ) -> List[int]:
        """Find indices for context expansion."""
        available_indices = list(chunks.keys())
        current_idx = int(doc["index"])
        included_indices = {current_idx}
        
        max_expansion = max(
            max(available_indices) - current_idx,
            current_idx - min(available_indices)
        )
        
        for k in range(1, max_expansion):
            expand_range = [current_idx + k, current_idx - k]
            included_indices, break_flag = self._expand_chunk_range(
                chunks, doc, expand_range, expand_len
            )
            if break_flag:
                break
        
        return sorted(list(included_indices))

    def expand_document(
        self, 
        doc: Dict[str, Any], 
        expand_len: int, 
        score: float, 
        result_queue: Queue
    ) -> None:
        """
        Expand a single document's context (runs in separate process).
        
        Args:
            doc: Document to expand
            expand_len: Maximum expansion length
            score: Document score
            result_queue: Queue to put result
        """
        try:
            chunk_path = doc.get("chunk_path")
            if not chunk_path or not os.path.exists(chunk_path):
                logger.warning(f"Chunk file not found: {chunk_path}")
                doc["score"] = score
                result_queue.put(doc)
                return
            
            # Load chunks from the same section
            with open(chunk_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            chunks = {}
            for line in lines:
                try:
                    chunk = json.loads(line)
                    if chunk["section"] == doc["section"]:
                        chunks[int(chunk["index"])] = chunk
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing chunk: {e}")
                    continue
            
            if int(doc["index"]) not in chunks:
                logger.warning(f"Document index {doc['index']} not found in chunks")
                doc["score"] = score
                result_queue.put(doc)
                return
            
            # Find expansion indices and build text
            expansion_indices = self._find_expansion_indices(chunks, doc, expand_len)
            
            expanded_text = ""
            for idx in expansion_indices:
                if idx in chunks:
                    expanded_text += chunks[idx]["text"]
            
            doc["text"] = expanded_text
            doc["score"] = score
            result_queue.put(doc)
            
        except Exception as e:
            logger.error(f"Error in document expansion: {e}")
            doc["score"] = score
            result_queue.put(doc)

    def query(
        self, 
        query: str, 
        k: int = TOP_K, 
        expand_len: int = EXPAND_RANGE, 
        score_threshold: float = SCORE_THRESHOLD
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Query the knowledge base and generate response.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            expand_len: Context expansion length
            score_threshold: Score threshold for filtering
            
        Returns:
            Tuple of (retrieved_documents, generated_response)
        """
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant documents
        docs = self.retrieval(query, k, expand_len, score_threshold)
        
        if not docs:
            logger.info("No relevant documents found, using LLM direct response")
            response = generate_llm_response(query)
        else:
            # Build context from retrieved documents
            docs_text = "\n".join([
                f"[document {idx} begin]{doc['text']}[document {idx} end]"
                for idx, doc in enumerate(docs)
            ])
            
            # Create prompt with context
            current_date = today()
            prompt = RAG_PROMPT.format(
                search_results=docs_text,
                cur_date=current_date,
                question=query
            )
            
            response = generate_llm_response(prompt)
        
        logger.info("Query processing completed")
        return docs, response
