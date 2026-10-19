import os
import sqlite3
import uuid
import warnings
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

DB_NAME = "rag_app.db"
CHROMA_DIR = "./chroma_db"
DOCS_DIR = "sample_documents"
COLLECTION_NAME = "rag_collection"


# Database Management (SQLite)


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    gpt_response TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def insert_application_log(session_id: str, user_query: str, gpt_response: str, model: str):
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
        (session_id, user_query, gpt_response, model)
    )
    conn.commit()
    conn.close()

def get_chat_history_messages(session_id: str) -> List[BaseMessage]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at ASC',
        (session_id,)
    )
    messages: List[BaseMessage] = []
    for row in cursor.fetchall():
        messages.append(HumanMessage(content=row['user_query']))
        messages.append(AIMessage(content=row['gpt_response']))
    conn.close()
    return messages

def get_chat_history_dicts(session_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id, session_id, user_query, gpt_response, model, created_at FROM application_logs WHERE session_id = ? ORDER BY created_at ASC',
        (session_id,)
    )
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return history

def get_all_sessions() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''SELECT session_id, COUNT(*) as message_count, MAX(created_at) as last_active, MIN(created_at) as started_at
           FROM application_logs
           GROUP BY session_id
           ORDER BY last_active DESC'''
    )
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return sessions

def delete_session_history(session_id: str) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM application_logs WHERE session_id = ?', (session_id,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


# Document processing & vector store

def get_embeddings():
    embedding_choice = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    if embedding_choice == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings()
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

def load_single_file(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html') or file_path.endswith('.htm'):
        from langchain_community.document_loaders import BSHTMLLoader
        loader = BSHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Only .pdf, .docx, .html, and .htm are supported.")
    return loader.load()

def load_documents_from_folder(folder_path: str = DOCS_DIR) -> List[Document]:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return []

    documents: List[Document] = []
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(('.pdf', '.docx', '.html', '.htm')):
            try:
                docs = load_single_file(file_path)
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return documents

# -------------------------------------------------------------
# Answer Cleaning & Post-Processing
# -------------------------------------------------------------

def clean_rag_answer(raw_answer: str, query: str = "") -> str:
    """
    Refines raw LLM outputs by removing leading question artifacts, repetitive phrases,
    meta-commentary, trailing pleasantries/filler, and incomplete truncated phrases.
    """
    if not raw_answer:
        return "The requested information is not available in the provided documents."

    import re
    text = raw_answer.strip()

    # If the response indicates missing info or refusal
    not_found_triggers = [
        "not available in the provided documents",
        "does not contain any relevant information",
        "does not provide any relevant information",
        "cannot determine what content is inside",
        "no relevant information",
        "is not available in the context",
        "not mentioned in the provided",
        "not mentioned in the context"
    ]
    lower_text = text.lower()
    if any(t in lower_text for t in not_found_triggers):
        if (
            "guidelines have been followed" in lower_text
            or "request was successful" in lower_text
            or "cannot determine" in lower_text
            or lower_text.count("not available") >= 2
            or len(text) < 180
        ):
            key_content_words = [
                "policy", "architecture", "security", "framework", "stipend", "leave",
                "hours", "encryption", "pricing", "tier", "refund", "standard", "employee"
            ]
            if not any(w in lower_text for w in key_content_words):
                return "The requested information is not available in the provided documents."

    # 1. Strip leading meta-thinking / preambles / question completions
    text = re.sub(
        r'^(Directly answering[^\n]+\n+|Instead,[^\n]+\n+|Here is the answer:\s*|Answer:\s*|\?\s*(Answer:)?\s*)+',
        '',
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r'^[?\s:,\-]+', '', text)
    text = re.sub(r'^(Answer|Response|Assistant|AI):\s*', '', text, flags=re.IGNORECASE)

    # 2. Strip trailing pleasantries, disclaimers, hashtags, and conversational filler
    fluff_patterns = [
        r'\[End of message\]',
        r'\[End of text\]',
        r'<\|im_end\|>',
        r'<\|endoftext\|>',
        r'#\w+',
        r'\n*Please let me know if you need more detailed information.*',
        r'\n*Please let me know if you (need|have|require).*',
        r'\n*Let me know (how else|if you).*',
        r'\n*If there\'s anything else.*',
        r'\n*If you (need|have) (more|any|additional).*',
        r'\n*Thank you for your understanding.*',
        r'\n*I\'m here to help!.*',
        r'\n*The guidelines have been followed.*',
        r'\n*Otherwise, my previous response meets all requirements.*',
        r'\n*Please feel free to ask.*'
    ]
    for pat in fluff_patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE | re.DOTALL)

    # 3. If there is a trailing duplicate 'The requested information is not available' appended to an actual answer
    if "The requested information is not available in the provided documents." in text and len(text) > 120:
        text = text.replace("The requested information is not available in the provided documents.", "").strip()

    # 4. Remove adjacent duplicate sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    deduped = []
    for s in sentences:
        if not deduped or s.lower() != deduped[-1].lower():
            deduped.append(s)
    text = ' '.join(deduped).strip()

    # 5. Fix truncated dangling sentences at the end (trim to last complete sentence)
    if text and text[-1] not in '.!?\"\'`':
        last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_punct != -1 and last_punct > len(text) * 0.4:
            text = text[:last_punct + 1].strip()

    return text if text else "The requested information is not available in the provided documents."

# -------------------------------------------------------------
# HuggingFace & Fallback LLM Setup
# -------------------------------------------------------------

_cached_hf_llm = None

def get_huggingface_llm():
    """
    Returns an optimized Hugging Face pipeline LLM.
    Uses Qwen/Qwen2.5-0.5B-Instruct with repetition penalty and tokenizer padding.
    """
    global _cached_hf_llm
    if _cached_hf_llm is not None:
        return _cached_hf_llm

    from transformers import pipeline, AutoTokenizer
    from langchain_huggingface import HuggingFacePipeline

    model_name = os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        max_new_tokens=256,
        return_full_text=False,
        do_sample=False,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )
    _cached_hf_llm = HuggingFacePipeline(pipeline=pipe)
    return _cached_hf_llm

class RAGPipeline:
    def __init__(self):
        init_db()
        self.embeddings = get_embeddings()
        self.text_splitter = get_text_splitter()
        self.vectorstore: Optional[Chroma] = None
        self.hf_llm = get_huggingface_llm()
        self.chains: Dict[str, Any] = {}
        self._init_vectorstore()

    def _init_vectorstore(self):
        """Initializes ChromaDB, indexing sample_documents if already present."""
        os.makedirs(CHROMA_DIR, exist_ok=True)
        
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR
        )

        existing_count = self.vectorstore._collection.count()
        if existing_count == 0:
            docs = load_documents_from_folder(DOCS_DIR)
            if docs:
                splits = self.text_splitter.split_documents(docs)
                self.vectorstore.add_documents(splits)
                print(f"[RAG] Indexed {len(splits)} initial document chunks into ChromaDB.")

        self._build_chains()

    def _build_chains(self):
        """Builds default RAG chain with HuggingFace as default and OpenAI with automatic fallback."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

        # 1. HuggingFace Default Chain with refined prompt
        hf_system_prompt = (
            "You are a factual, concise document search assistant. Answer the user's question directly "
            "based ONLY on the provided context below.\n\n"
            "Rules:\n"
            "- State the facts directly without preamble, pleasantries, hashtags, or conversational filler.\n"
            "- Do not repeat the question.\n"
            "- If the answer is not in the context, reply: 'The requested information is not available in the provided documents.'\n\n"
            "Context:\n{context}"
        )
        hf_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", hf_system_prompt),
            ("human", "Question: {input}\nDirect Answer:")
        ])
        hf_qa_chain = create_stuff_documents_chain(self.hf_llm, hf_qa_prompt)
        self.chains["huggingface"] = create_retrieval_chain(retriever, hf_qa_chain)

        # 2. OpenAI Chain with HuggingFace Fallback
        if os.getenv("OPENAI_API_KEY"):
            try:
                openai_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                # True LangChain fallback: if OpenAI throws 429 quota or connection error, fallback to HF
                llm_with_fallback = openai_llm.with_fallbacks([self.hf_llm])

                contextualize_q_system_prompt = (
                    "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, "
                    "just reformulate it if needed and otherwise return as it is."
                )
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                history_aware_retriever = create_history_aware_retriever(
                    llm_with_fallback, retriever, contextualize_q_prompt
                )

                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a professional AI Assistant for company document search. "
                               "Answer the user's question accurately based ONLY on the provided context below. "
                               "If the answer cannot be found in the context, politely state that the information "
                               "is not available in the company documents.\n\nContext:\n{context}"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])
                question_answer_chain = create_stuff_documents_chain(llm_with_fallback, qa_prompt)
                self.chains["openai"] = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            except Exception as e:
                print(f"[RAG] Could not configure OpenAI chain: {e}")

    def add_file(self, file_path: str) -> int:
        docs = load_single_file(file_path)
        splits = self.text_splitter.split_documents(docs)
        if splits:
            self.vectorstore.add_documents(splits)
        return len(splits)

    def list_documents(self) -> List[str]:
        if not os.path.exists(DOCS_DIR):
            return []
        return sorted([
            f for f in os.listdir(DOCS_DIR)
            if f.endswith(('.pdf', '.docx', '.html', '.htm'))
        ])

    def delete_document(self, filename: str) -> bool:
        file_path = os.path.join(DOCS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        self.reindex_all(DOCS_DIR)
        return True

    def reindex_all(self, folder_path: str = DOCS_DIR) -> int:
        docs = load_documents_from_folder(folder_path)
        splits = self.text_splitter.split_documents(docs)
        
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR
        )
        if splits:
            self.vectorstore.add_documents(splits)
        self._build_chains()
        return len(splits)

    def ask(self, query: str, session_id: Optional[str] = None, model: str = "huggingface") -> Dict[str, Any]:
        if not session_id:
            session_id = str(uuid.uuid4())

        chat_history = get_chat_history_messages(session_id)
        
        # Decide which chain to use
        is_openai_requested = model and "gpt" in model.lower() and "openai" in self.chains
        active_chain = self.chains["openai"] if is_openai_requested else self.chains["huggingface"]
        effective_model = model if is_openai_requested else "Qwen/Qwen2.5-0.5B-Instruct (HuggingFace)"

        # Contextual augmentation for follow-up queries with pronouns in local model
        effective_query = query
        if not is_openai_requested and chat_history:
            followup_pronouns = {"it", "its", "they", "them", "their", "this", "that", "these", "those"}
            query_words = set(query.lower().replace("?", "").replace(".", "").split())
            if query_words & followup_pronouns:
                prev_human_msgs = [m.content for m in chat_history if isinstance(m, HumanMessage)]
                if prev_human_msgs:
                    effective_query = f"{prev_human_msgs[-1]}. {query}"

        try:
            response = active_chain.invoke({
                "input": effective_query,
                "chat_history": chat_history
            })
            raw_answer = response["answer"].strip()
            answer = clean_rag_answer(raw_answer, query=query)
            retrieved_docs = response.get("context", [])

            sources = []
            for doc in retrieved_docs:
                src = doc.metadata.get("source", "Unknown")
                sources.append({
                    "source": os.path.basename(src),
                    "snippet": doc.page_content[:300].strip()
                })

            insert_application_log(session_id, query, answer, effective_model)

            return {
                "session_id": session_id,
                "query": query,
                "answer": answer,
                "sources": sources,
                "model_used": effective_model
            }

        except Exception as e:
            # Automatic graceful fallback to HuggingFace if primary chain failed
            print(f"[RAG Fallback] Primary chain failed with '{e}'. Executing HuggingFace fallback...")
            try:
                hf_chain = self.chains["huggingface"]
                fallback_res = hf_chain.invoke({"input": effective_query, "chat_history": []})
                raw_fallback = fallback_res["answer"].strip()
                answer = clean_rag_answer(raw_fallback, query=query)
                retrieved_docs = fallback_res.get("context", [])

                sources = []
                for doc in retrieved_docs:
                    src = doc.metadata.get("source", "Unknown")
                    sources.append({
                        "source": os.path.basename(src),
                        "snippet": doc.page_content[:300].strip()
                    })

                fallback_note = f"{answer}\n\n*(Generated via local HuggingFace fallback due to upstream provider error: {str(e)[:70]}...)*"
                insert_application_log(session_id, query, fallback_note, "Qwen/Qwen2.5-0.5B-Instruct (HuggingFace Fallback)")

                return {
                    "session_id": session_id,
                    "query": query,
                    "answer": fallback_note,
                    "sources": sources,
                    "model_used": "Qwen/Qwen2.5-0.5B-Instruct (HuggingFace Fallback)"
                }
            except Exception as hf_err:
                raise RuntimeError(f"Both primary and HuggingFace fallback failed: {hf_err}")

# Global singleton pipeline instance
rag_pipeline = RAGPipeline()
