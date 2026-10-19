import os
import shutil
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_service import (
    rag_pipeline,
    get_all_sessions,
    get_chat_history_dicts,
    delete_session_history,
    DOCS_DIR
)


# FastAPI application configuration


app = FastAPI(
    title="Conversational RAG Chatbot API",
    description="Production-ready FastAPI backend for LangChain RAG pipeline with ChromaDB and SQLite multi-user history persistence.",
    version="1.0.0"
)

# enable CORS for Streamlit and external web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Request & Response Schemas


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user query or question", min_length=1)
    session_id: Optional[str] = Field(None, description="Optional conversation session ID (UUID). Generates a new one if omitted.")
    model: Optional[str] = Field("huggingface", description="LLM model identifier: 'huggingface' (default local free) or 'gpt-4o-mini'")

class SourceItem(BaseModel):
    source: str
    snippet: str

class ChatResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    sources: List[SourceItem]
    model_used: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    indexed_documents_count: int
    active_collection: str
    embedding_model: str
    llm_model: str
    storage_directory: str

class SessionSummary(BaseModel):
    session_id: str
    message_count: int
    last_active: str
    started_at: str

# api endpoints


@app.get("/", tags=["System"])
def root():
    return {
        "name": "Conversational RAG Chatbot API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "status": "operational"
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns system status, total indexed document chunks, and embedding configuration."""
    count = rag_pipeline.vectorstore._collection.count() if rag_pipeline.vectorstore else 0
    return HealthResponse(
        status="healthy",
        indexed_documents_count=count,
        active_collection="rag_collection",
        embedding_model="all-MiniLM-L6-v2 (SentenceTransformers)",
        llm_model="Qwen/Qwen2.5-0.5B-Instruct (HuggingFace Default)",
        storage_directory="./chroma_db"
    )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Primary endpoint for RAG conversational search.
    - Contextualizes follow-up questions using past chat turns from SQLite.
    - Retrieves top-k relevant document chunks from ChromaDB.
    - Generates grounded response using default HuggingFace model or OpenAI with automatic fallback.
    """
    try:
        result = rag_pipeline.ask(
            query=request.message,
            session_id=request.session_id,
            model=request.model or "huggingface"
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating RAG response: {str(e)}"
        )

@app.get("/chat/sessions", response_model=List[SessionSummary], tags=["Sessions"])
def list_sessions():
    """Lists all user sessions with their message count and timestamps."""
    return get_all_sessions()

@app.get("/chat/history/{session_id}", tags=["Sessions"])
def get_session_history(session_id: str):
    """Retrieves full conversation history for a given session ID."""
    history = get_chat_history_dicts(session_id)
    if not history:
        return {"session_id": session_id, "messages": [], "total": 0}
    return {
        "session_id": session_id,
        "messages": history,
        "total": len(history)
    }

@app.delete("/chat/history/{session_id}", tags=["Sessions"])
def clear_session_history(session_id: str):
    """Deletes conversation history for a specific session ID."""
    deleted_count = delete_session_history(session_id)
    return {
        "session_id": session_id,
        "deleted_records": deleted_count,
        "status": "cleared"
    }

ALLOWED_EXTENSIONS = ('.pdf', '.docx', '.html', '.htm')

@app.get("/documents", tags=["Documents"])
def get_documents():
    """List all documents currently available in sample_documents/."""
    return {"documents": rag_pipeline.list_documents()}

@app.delete("/documents/{filename}", tags=["Documents"])
def delete_document(filename: str):
    """Delete a document and reindex the vector store."""
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document filename."
        )
    rag_pipeline.delete_document(filename)
    return {
        "filename": filename,
        "status": "deleted",
        "total_indexed_chunks": rag_pipeline.vectorstore._collection.count()
    }

@app.post("/documents/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a new .pdf, .docx, .html, or .htm document, save to sample_documents/,
    and dynamically index into ChromaDB vector store.
    """
    filename = file.filename
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a .pdf, .docx, .html, or .htm document."
        )

    os.makedirs(DOCS_DIR, exist_ok=True)
    destination_path = os.path.join(DOCS_DIR, filename)

    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # index into ChromaDB
        chunks_added = rag_pipeline.add_file(destination_path)
        total_chunks = rag_pipeline.vectorstore._collection.count()

        return {
            "filename": filename,
            "status": "indexed",
            "chunks_added": chunks_added,
            "total_indexed_chunks": total_chunks
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index document: {str(e)}"
        )

@app.post("/documents/reindex", tags=["Documents"])
def reindex_documents():
    """Reindexes all files found in sample_documents/."""
    try:
        total_splits = rag_pipeline.reindex_all(DOCS_DIR)
        return {
            "status": "reindexed",
            "total_chunks": total_splits,
            "source_directory": DOCS_DIR
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reindex: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
