"""
FastAPI application for GeoGPT-RAG pipeline.

Provides REST endpoints for document ingestion, querying, and knowledge base management.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .kb import KBDocQA
from .config import UPLOAD_DIR, MILVUS_COLLECTION


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for querying the knowledge base."""
    query: str = Field(..., description="The question to ask")
    k: int = Field(3, description="Number of documents to retrieve", ge=1, le=20)
    expand_len: int = Field(1024, description="Context expansion length", ge=0, le=4096)
    score_threshold: float = Field(1.5, description="Score threshold for filtering", ge=0.0)


class QueryResponse(BaseModel):
    """Response model for query results."""
    query: str
    response: str
    documents: List[Dict[str, Any]]
    retrieval_count: int


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    filename: str
    status: str
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    collection: str
    components: Dict[str, str]


# Global KB instance
kb_instance: Optional[KBDocQA] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global kb_instance
    logging.info("Initializing GeoGPT-RAG application...")
    
    try:
        kb_instance = KBDocQA()
        logging.info("KBDocQA instance initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize KBDocQA: {e}")
        raise RuntimeError(f"Application startup failed: {e}")
    
    yield
    
    # Shutdown
    logging.info("Shutting down GeoGPT-RAG application...")


# FastAPI app
app = FastAPI(
    title="GeoGPT-RAG API",
    description="Retrieval-Augmented Generation pipeline for geographic and scientific documents",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "GeoGPT-RAG API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if kb_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KB instance not initialized"
        )
    
    return HealthResponse(
        status="healthy",
        collection=MILVUS_COLLECTION,
        components={
            "embeddings": "operational",
            "reranking": "operational", 
            "vector_store": "operational",
            "text_splitter": "operational"
        }
    )


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to upload")
):
    """
    Upload and process a document into the knowledge base.
    
    Supported formats: .txt, .md, .pdf (text-based)
    """
    if kb_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KB instance not initialized"
        )
    
    # Validate file type
    allowed_extensions = {".txt", ".md", ".pdf"}
    file_extension = Path(file.filename or "").suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_extension} not supported. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / (file.filename or "upload.txt")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process file in background
        background_tasks.add_task(process_document_background, str(file_path))
        
        return DocumentUploadResponse(
            filename=file.filename or "unknown",
            status="processing",
            chunks_created=0,  # Will be updated after processing
            message="Document uploaded successfully and is being processed"
        )
        
    except Exception as e:
        logging.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


async def process_document_background(file_path: str):
    """Background task to process uploaded document."""
    try:
        if kb_instance:
            kb_instance.add_file(file_path)
            logging.info(f"Successfully processed document: {file_path}")
    except Exception as e:
        logging.error(f"Error processing document {file_path}: {e}")


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base with a question.
    
    Returns relevant documents and a generated response.
    """
    if kb_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KB instance not initialized"
        )
    
    try:
        # Query the knowledge base
        documents, response = kb_instance.query(
            query=request.query,
            k=request.k,
            expand_len=request.expand_len,
            score_threshold=request.score_threshold
        )
        
        return QueryResponse(
            query=request.query,
            response=response,
            documents=documents,
            retrieval_count=len(documents)
        )
        
    except Exception as e:
        logging.error(f"Error querying knowledge base: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying knowledge base: {str(e)}"
        )


@app.post("/retrieve", response_model=Dict[str, Any])
async def retrieve_documents(request: QueryRequest):
    """
    Retrieve relevant documents without LLM generation.
    
    Useful for debugging or when you only need document retrieval.
    """
    if kb_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KB instance not initialized"
        )
    
    try:
        # Retrieve documents only
        documents = kb_instance.retrieval(
            query=request.query,
            k=request.k,
            expand_len=request.expand_len,
            score_threshold=request.score_threshold
        )
        
        return {
            "query": request.query,
            "documents": documents,
            "count": len(documents)
        }
        
    except Exception as e:
        logging.error(f"Error retrieving documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving documents: {str(e)}"
        )


@app.delete("/collection")
async def drop_collection():
    """
    Drop the current vector collection.
    
    ⚠️ Warning: This will delete all stored documents!
    """
    if kb_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KB instance not initialized"
        )
    
    try:
        kb_instance.drop_collection()
        return {"message": f"Collection '{MILVUS_COLLECTION}' dropped successfully"}
        
    except Exception as e:
        logging.error(f"Error dropping collection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error dropping collection: {str(e)}"
        )


@app.get("/stats")
async def get_stats():
    """Get basic statistics about the knowledge base."""
    if kb_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KB instance not initialized"
        )
    
    try:
        # Basic stats - could be expanded
        return {
            "collection_name": MILVUS_COLLECTION,
            "upload_directory": str(UPLOAD_DIR),
            "status": "operational"
        }
        
    except Exception as e:
        logging.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )


if __name__ == "__main__":
    # For development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
