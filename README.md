# Offline Personal AI Assistant with RAG

This is a 100% offline personal AI assistant that uses Retrieval-Augmented Generation (RAG) to answer questions based on your own personal data. It now features both a **Streamlit Web UI** and a **Voice Interaction** mode.

## Features
- **Brain:** Ollama (Default: Gemma 3:4b, supports Llama 3, Phi 3, etc.).
- **RAG Engine:** Pure Python vector search using `sentence-transformers` and `numpy`.
- **Personal Data:** Automatically indexes PDFs, TXT, and DOCX files from the `data/` folder.
- **Interface:** Beautiful Streamlit Web UI with chat history and source transparency.
- **Voice Mode:** Optional STT/TTS mode using Faster-Whisper and pyttsx3.

## Prerequisites
1. **Ollama:** Download from [ollama.com](https://ollama.com) and run `ollama pull gemma3:4b`.
2. **Python 3.9+**
3. **Data:** Place your PDF, TXT, and DOCX files in the `offline_assistant/data/` folder.

## Setup
1. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Assistant

### Option 1: Streamlit Web UI (Recommended)
```bash
streamlit run streamlit_app.py
```

### Option 2: Voice Interaction Mode
```bash
python assistant.py
```

## How to Use
- **Web UI:** Type your questions in the chat box. The assistant will search your documents and display sources.
- **Reindexing:** Use the "Reindex Personal Data" button in the sidebar if you add or change files in the `data/` folder.
- **Privacy:** Everything stays on your machine. No data is sent to the cloud.

## Technical Details
- **Embeddings:** `all-MiniLM-L6-v2` (Local).
- **Vector Store:** Custom `pickle` + `numpy` implementation.
- **System Prompt:** Instructs the LLM to prioritize retrieved personal context.
