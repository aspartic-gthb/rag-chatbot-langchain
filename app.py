import os
import uuid
import streamlit as st
from rag_service import (
    rag_pipeline,
    DOCS_DIR,
    get_chat_history_messages
)

# -------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------

st.set_page_config(
    page_title="Langchain RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

#Sidebar: Model Selection, Upload, Document Management

# Select Model
model_dict = {
    "Qwen/Qwen2.5-0.5B-Instruct (HuggingFace)": "huggingface",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o"
}
selected_model_label = st.sidebar.selectbox(
    "Select Model",
    options=list(model_dict.keys()),
    index=0
)
selected_model = model_dict[selected_model_label]

#Upload Document
st.sidebar.markdown("### Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=["pdf", "docx", "html", "htm"]
)

# Auto index immediately upon file upload
if uploaded_file is not None:
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = set()

    file_sig = f"{uploaded_file.name}_{uploaded_file.size}"
    if file_sig not in st.session_state.indexed_files:
        os.makedirs(DOCS_DIR, exist_ok=True)
        file_path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with st.sidebar.status(f"Auto-indexing {uploaded_file.name}..."):
            chunks_added = rag_pipeline.add_file(file_path)

        st.session_state.indexed_files.add(file_sig)
        st.sidebar.success(f"Uploaded and indexed {uploaded_file.name} ({chunks_added} chunks)")
        st.rerun()

if st.sidebar.button("Upload"):
    if uploaded_file is not None:
        file_path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        with st.sidebar.status(f"Indexing {uploaded_file.name}..."):
            chunks_added = rag_pipeline.add_file(file_path)
        st.sidebar.success(f"Uploaded and indexed {uploaded_file.name} ({chunks_added} chunks)")
        st.rerun()
    else:
        st.sidebar.warning("Please choose a file to upload.")

# uploaded Documents
st.sidebar.markdown("### Uploaded Documents")

if st.sidebar.button("Refresh Document List"):
    st.rerun()

# list uploaded files
docs_list = rag_pipeline.list_documents()

for doc in docs_list:
    st.sidebar.text(doc)

# document deletion
selected_doc = st.sidebar.selectbox(
    "Select a document to delete",
    options=docs_list if docs_list else ["No documents available"]
)

if st.sidebar.button("Delete Selected Document"):
    if selected_doc and selected_doc != "No documents available":
        with st.sidebar.status(f"Deleting {selected_doc}..."):
            rag_pipeline.delete_document(selected_doc)
        if "indexed_files" in st.session_state:
            st.session_state.indexed_files = {
                sig for sig in st.session_state.indexed_files if not sig.startswith(selected_doc + "_")
            }
        st.sidebar.success(f"Deleted {selected_doc}")
        st.rerun()


# Main chatt Area


st.markdown("<h1 style='text-align: center; color: #262730; font-weight: 700; margin-bottom: 2rem;'>Langchain RAG Chatbot</h1>", unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("details"):
            with st.expander("Details", expanded=False):
                st.write(msg["details"])

# Chat Input with "Query:" placeholder matching the screenshot
user_query = st.chat_input("Query:")

if user_query:
    # 1. Retrieve relevant context / chunks before generation
    retrieved_chunks = []
    details_content = ""
    try:
        if rag_pipeline.vectorstore:
            search_query = user_query
            followup_pronouns = {"it", "its", "they", "them", "their", "this", "that", "these", "those"}
            query_words = set(user_query.lower().replace("?", "").replace(".", "").split())
            if query_words & followup_pronouns:
                prev_user_queries = [m["content"] for m in st.session_state.messages if m.get("role") == "user"]
                if prev_user_queries:
                    search_query = f"{prev_user_queries[-1]}. {user_query}"

            docs = rag_pipeline.vectorstore.similarity_search(search_query, k=2)
            if docs:
                retrieved_chunks = [
                    f"Source: {os.path.basename(d.metadata.get('source', 'Document'))}\n\n{d.page_content.strip()}"
                    for d in docs
                ]
                details_content = "\n\n---\n\n".join(retrieved_chunks)
    except Exception:
        pass

    # 2. Append user message to state
    user_message_entry = {
        "role": "user",
        "content": user_query,
        "details": details_content
    }
    st.session_state.messages.append(user_message_entry)

    # 3. Display user message immediately with Details expander
    with st.chat_message("user"):
        st.write(user_query)
        if details_content:
            with st.expander("Details", expanded=False):
                st.write(details_content)

    # 4. Generate assistant response with spinner
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                res = rag_pipeline.ask(
                    query=user_query,
                    session_id=st.session_state.session_id,
                    model=selected_model
                )
                answer = res.get("answer", "No response generated.")
            except Exception as e:
                answer = f"Error generating response: {str(e)}"

        st.write(answer)

    # 5. Append assistant message to state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
