# Document Question Answering System (RAG)

This project implements a simple Retrieval-Augmented Generation (RAG) system that answers questions based on uploaded custom documents. 

Instead of relying only on a language model's internal knowledge, the system retrieves relevant chunks of text from documents and then generates answers grounded in that information.

## Project Structure

*   `rag_system.py`: Core RAG pipeline containing:
    *   Document ingestion (PDF and Text parsing using `pypdf`)
    *   Text chunking (Splitting text into overlapping segments)
    *   Embedding creation (Using Gemini API `text-embedding-004`)
    *   Similarity search (Calculating manually with Cosine Similarity using NumPy)
    *   Answer generation (Using `gemini-1.5-flash` model)
*   `app.py`: Streamlit-based web interface for uploading documents and asking questions.
*   `requirements.txt`: Python package dependencies list.

## Setup & Installation

### 1. Install Dependencies
Run the following command to install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Make sure you have your Gemini API Key. You can set it as an environment variable:
```bash
export GEMINI_API_KEY="your_api_key_here"
```
Or you can paste it directly into the web interface or terminal when prompted.

## How to Run

### Option 1: Command Line Interface (CLI)
You can run the pipeline directly in your console:
```bash
python rag_system.py
```
This script will ask you for a document file path, index it, and open an interactive QA loop.

### Option 2: Streamlit Web UI
You can run the web-based app:
```bash
streamlit run app.py
```
This starts the local web server at `http://localhost:8501`. You can upload a PDF/TXT document, ask questions, and see the retrieved context sections along with their similarity scores.
