"""
FastAPI application for GeoGPT-RAG pipeline.

Provides REST endpoints for document ingestion, querying, and knowledge base management.
"""

from __future__ import annotations

# Apply compatibility fixes before any other imports
import os
import warnings

# Suppress TensorFlow and transformers warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set compatibility environment variables
compatibility_env_vars = {
    'TOKENIZERS_PARALLELISM': 'false',
    'TRANSFORMERS_OFFLINE': '0',
    'HF_HUB_DISABLE_TELEMETRY': '1',
    'CUDA_LAUNCH_BLOCKING': '0',
}

for key, value in compatibility_env_vars.items():
    if key not in os.environ:
        os.environ[key] = value

# Apply MistralDualModel compatibility patch
def apply_mistral_dual_model_patch():
    """Apply compatibility patch for MistralDualModel _update_causal_mask issue."""
    try:
        from transformers import PreTrainedModel
        
        def _update_causal_mask_patch(self, attention_mask=None, input_ids=None, **kwargs):
            """Compatibility patch for _update_causal_mask method."""
            # For embedding models, we don't need causal masking
            return None
        
        # Apply the patch globally
        if not hasattr(PreTrainedModel, '_update_causal_mask'):
            PreTrainedModel._update_causal_mask = _update_causal_mask_patch
            print("✅ Applied MistralDualModel compatibility patch")
            
    except Exception as e:
        print(f"⚠️  Could not apply MistralDualModel patch: {e}")

# Apply the patch immediately
apply_mistral_dual_model_patch()

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
from .models.sagemaker_llm import SageMakerLLMClient
from .config import (
    UPLOAD_DIR, 
    MILVUS_COLLECTION, 
    LLM_PROVIDER, 
    LLM_URL, 
    LLM_KEY, 
    SAGEMAKER_ENDPOINT_NAME, 
    AWS_REGION
)


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
        # Ensure upload directory exists
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Clean and secure the filename
        original_filename = file.filename or "upload.txt"
        # Remove any path components and ensure it's just a filename
        safe_filename = Path(original_filename).name
        if not safe_filename:
            safe_filename = "upload.txt"
        
        # Save uploaded file
        file_path = UPLOAD_DIR / safe_filename
        
        # Read file content first
        content = await file.read()
        
        # Check if file is empty
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
        
        # Write file to disk
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Process file in background
        background_tasks.add_task(process_document_background, str(file_path))
        
        return DocumentUploadResponse(
            filename=safe_filename,
            status="processing",
            chunks_created=0,  # Will be updated after processing
            message="Document uploaded successfully and is being processed"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like empty file)
        raise
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
            logging.info(f"Starting background processing of document: {file_path}")
            
            # Check if file exists and is readable
            if not os.path.exists(file_path):
                logging.error(f"File not found for processing: {file_path}")
                return
            
            # Check file permissions
            if not os.access(file_path, os.R_OK):
                logging.error(f"File not readable for processing: {file_path}")
                return
            
            file_size = os.path.getsize(file_path)
            logging.info(f"Processing file: {file_path} (size: {file_size} bytes)")
            
            # Process the file
            kb_instance.add_file(file_path)
            logging.info(f"Successfully processed document: {file_path}")
            
            # Clean up the uploaded file after processing
            try:
                os.remove(file_path)
                logging.info(f"Cleaned up uploaded file: {file_path}")
            except Exception as cleanup_error:
                logging.warning(f"Could not clean up file {file_path}: {cleanup_error}")
                
        else:
            logging.error("KB instance not available for document processing")
            
    except Exception as e:
        logging.error(f"Error processing document {file_path}: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")


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
            "retrieval_count": len(documents)
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
        return {
            "message": "Collection dropped successfully",
            "collection": MILVUS_COLLECTION
        }
        
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
        # Get document count from vector store if available
        document_count = 0
        stats_status = "operational"
        
        try:
            if hasattr(kb_instance, 'vector_store') and kb_instance.vector_store and hasattr(kb_instance.vector_store, 'col'):
                if kb_instance.vector_store.col is not None:
                    document_count = kb_instance.vector_store.col.num_entities
                else:
                    stats_status = "no_collection"
        except Exception:
            # If we can't get the count, default to 0
            pass
        
        return {
            "collection_name": MILVUS_COLLECTION,
            "document_count": document_count,
            "upload_directory": str(UPLOAD_DIR),
            "llm_provider": LLM_PROVIDER,
            "status": stats_status
        }
        
    except Exception as e:
        logging.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )


@app.get("/llm/test")
async def test_llm_connection():
    """Test LLM connection and configuration."""
    try:
        if LLM_PROVIDER.lower() == "sagemaker":
            # Test SageMaker endpoint
            if not SAGEMAKER_ENDPOINT_NAME:
                return {
                    "status": "error",
                    "provider": "sagemaker",
                    "error": "SAGEMAKER_ENDPOINT_NAME not configured"
                }
            
            try:
                sagemaker_client = SageMakerLLMClient(
                    endpoint_name=SAGEMAKER_ENDPOINT_NAME,
                    region_name=AWS_REGION
                )
                test_result = sagemaker_client.test_connection()
                return {
                    "provider": "sagemaker",
                    **test_result
                }
            except Exception as e:
                return {
                    "status": "error",
                    "provider": "sagemaker",
                    "endpoint_name": SAGEMAKER_ENDPOINT_NAME,
                    "region": AWS_REGION,
                    "error": str(e)
                }
        
        else:
            # Test OpenAI-compatible endpoint
            if not LLM_URL or not LLM_KEY:
                return {
                    "status": "error",
                    "provider": "openai-compatible",
                    "error": "LLM_URL or LLM_KEY not configured"
                }
            
            try:
                import openai
                client = openai.Client(base_url=LLM_URL, api_key=LLM_KEY)
                models = client.models.list()
                
                if not models.data:
                    return {
                        "status": "error",
                        "provider": "openai-compatible",
                        "url": LLM_URL,
                        "error": "No models available"
                    }
                
                # Test with a simple prompt
                model = models.data[0].id
                test_response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hello, this is a test."}],
                    max_tokens=50,
                    temperature=0.1
                )
                
                return {
                    "status": "success",
                    "provider": "openai-compatible",
                    "url": LLM_URL,
                    "available_models": [model.id for model in models.data[:5]],  # Show first 5 models
                    "test_response": test_response.choices[0].message.content
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "provider": "openai-compatible",
                    "url": LLM_URL,
                    "error": str(e)
                }
        
    except Exception as e:
        logging.error(f"Error testing LLM connection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing LLM connection: {str(e)}"
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
