# LangChain RAG Chatbot

A multi-turn conversational Retrieval-Augmented Generation (RAG) assistant built with LangChain, ChromaDB, HuggingFace, and OpenAI. It features an interactive Streamlit UI and a production-ready FastAPI REST backend.

---

## Features

- **Document Ingestion & Auto-Indexing**: Upload PDF, DOCX, HTML, or HTM files; documents are automatically chunked and indexed into ChromaDB.
- **Dual LLM Support**:
  - **HuggingFace (Default)**: Runs locally using `Qwen/Qwen2.5-0.5B-Instruct` (free, fast, no API key required).
  - **OpenAI (Optional)**: Supports `gpt-4o-mini` and `gpt-4o` with automatic HuggingFace fallback if quota or network errors occur.
- **Multi-Turn Conversational Memory**: Chat turns and session logs are persistently stored in SQLite (`rag_app.db`). Follow-up questions maintain context across turns.
- **Refined, Grounded Responses**: Custom prompt engineering and post-processing eliminate leading artifacts, repetitive phrases, and conversational filler.
- **Dual Interface**:
  - **Streamlit Web App**: Clean, native UI with collapsible source inspection, auto-upload, and document deletion.
  - **FastAPI REST API**: High-performance endpoints for chat, session history, document upload, and health checks.

---

## Project Structure

```text
.
├── app.py                     # Streamlit frontend UI
├── main.py                    # FastAPI REST API backend
├── rag_service.py             # Core RAG pipeline, loaders, Chroma vectorstore, and SQLite logging
├── sample_documents/          # Ingested documents (PDF, DOCX, HTML, HTM)
├── chroma_db/                 # Persistent Chroma vector database
├── rag_app.db                 # SQLite database for conversation history & logs
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Installation & Setup

### 1. Clone the Repository & Set Up Virtual Environment

```bash
git clone https://github.com/aspartic-gthb/rag-chatbot-langchain.git
cd langchain

python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```env
# Optional: Set OpenAI key for GPT models (defaults to local HuggingFace if omitted)
OPENAI_API_KEY="your-openai-api-key"

# Optional: LangSmith Tracing
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your-langchain-api-key"
LANGCHAIN_PROJECT="RAG-Chatbot"

# Optional: Default HuggingFace Model
HF_MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
```

---

## Running the Application

### Option A: Streamlit UI (Recommended for Testing & Demos)

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

- **Model Selection**: Select between `Qwen/Qwen2.5-0.5B-Instruct (HuggingFace)` (default) or OpenAI models.
- **Upload Document**: Drag & drop or browse a `.pdf`, `.docx`, `.html`, or `.htm` file. It will automatically index into ChromaDB.
- **Ask Questions**: Ask questions against your indexed documents. Click **Details** under any query to inspect retrieved source chunks.

### Option B: FastAPI Backend

Start the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

#### Key API Endpoints

| Method   | Endpoint                     | Description                                        |
| :------- | :--------------------------- | :------------------------------------------------- |
| `GET`    | `/health`                    | System health and indexed chunk count              |
| `POST`   | `/chat`                      | RAG conversational search with session persistence |
| `GET`    | `/chat/sessions`             | List all conversation sessions                     |
| `GET`    | `/chat/history/{session_id}` | Retrieve history for a session                     |
| `DELETE` | `/chat/history/{session_id}` | Clear conversation history                         |
| `GET`    | `/documents`                 | List all available documents                       |
| `POST`   | `/documents/upload`          | Upload and dynamically index a new document        |
| `DELETE` | `/documents/{filename}`      | Delete a document and reindex ChromaDB             |

---

## Architecture & Pipeline Flow

```text
User Question / Uploaded File
             │
             ▼
┌──────────────────────────┐
│      Document Ingestion   │ ──► PyPDFLoader / Docx2txtLoader / BSHTMLLoader
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ RecursiveCharacterSplitter│ ──► 1000 chunk size, 200 overlap
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│   SentenceTransformers   │ ──► all-MiniLM-L6-v2 embeddings
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│        ChromaDB          │ ──► Vector storage & top-k similarity retrieval
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│   RAG Chain & Refinement  │ ──► Qwen2.5-0.5B-Instruct / GPT-4o-mini
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  SQLite Session Logging  │ ──► Persisted into rag_app.db
└──────────────────────────┘
```

---
