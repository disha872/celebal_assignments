import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from rag_system import SimpleRAGSystem

# Load environment variables (e.g. from .env file)
load_dotenv(override=True)

# Title of the application
st.title("Document Question Answering System (RAG)")

# Read API Key from the environment variable
api_key = os.environ.get("GEMINI_API_KEY", "")

# Settings Sidebar (Only parameters, no API Key inputs)
st.sidebar.header("RAG Configuration")
chunk_size = st.sidebar.slider("Chunk Size (Characters)", min_value=100, max_value=1000, value=500, step=50)
chunk_overlap = st.sidebar.slider("Chunk Overlap (Characters)", min_value=0, max_value=200, value=100, step=10)
k_retrieved = st.sidebar.slider("Retrieve Top-K Chunks", min_value=1, max_value=5, value=3, step=1)

# Display running mode in sidebar
if not api_key:
    st.sidebar.info("ℹ️ Running in **Offline Demo Mode** (GEMINI_API_KEY environment variable not set).")
else:
    st.sidebar.success("🟢 Running in **Online Mode** (Using Gemini API).")

# Initialize Session State for RAG System
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None

# Document Ingestion Section
st.header("1. Ingest Document")
uploaded_file = st.file_uploader("Upload a PDF or TXT document:", type=["pdf", "txt"])

if uploaded_file:
    # Index the file
    if st.session_state.indexed_filename != uploaded_file.name:
        with st.spinner("Processing document and generating vector index..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name
            
            try:
                # Initialize system (uses environment API key, defaults to "no_key" if empty)
                current_key = api_key if api_key else "no_key"
                rag = SimpleRAGSystem(current_key)
                rag.index_document(temp_path, chunk_size, chunk_overlap)
                
                # Store in session state
                st.session_state.rag_system = rag
                st.session_state.indexed_filename = uploaded_file.name
                
                if current_key == "no_key":
                    st.warning("⚠️ Document indexed in Offline Mode (no GEMINI_API_KEY environment variable found).")
                else:
                    st.success(f"Successfully indexed document: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Failed to process document: {str(e)}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

# Question Answering Section
if st.session_state.rag_system:
    st.header("2. Ask Questions")
    st.write(f"Currently querying: `{st.session_state.indexed_filename}`")
    
    # Ensure the system uses the latest API key if loaded later
    st.session_state.rag_system.api_key = api_key if api_key else "no_key"
    
    query = st.text_input("Enter your question:")
    
    if st.button("Get Answer") and query:
        with st.spinner("Searching and generating response..."):
            try:
                answer, retrieved_chunks = st.session_state.rag_system.ask_question(query, k=k_retrieved)
                
                # Display Answer
                st.subheader("Answer:")
                st.write(answer)
                
                # Display Retrieved Context
                st.subheader("Retrieved Context Chunks:")
                for i, (score, chunk) in enumerate(retrieved_chunks):
                    st.markdown(f"**Chunk {i+1}** (Similarity Score: `{score:.4f}`):")
                    st.info(chunk)
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("Upload and index a document first to enable the Q&A interface.")
