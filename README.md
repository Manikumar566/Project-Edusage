# EduSage (Updated) — Q&A + Summarize + Generate Knowledge bites

## What’s new
- ✅ Core Q&A unchanged (paragraph answers, multi‑question history)
- ✅ Sidebar extras: **Summarize PDF(s)** and **Generate Knowledge bites**
- ✅ Windows‑safe file handling (no temp file lock)
- ✅ Outputs auto‑cleared on app start
- ✅ Uses Cohere API via `.env`

## Setup
```bash
pip install -r requirements.txt
 # on Windows
# then open .env and paste your Cohere key
# COHERE_API_KEY=your_key_here
streamlit run app.py
```

## Files
- `app.py` — Streamlit UI (Q&A + extras)
- `qa.py` — Cohere chat, embeddings, retrieval, summarization, flashcards
- `utils.py` — PDF text extraction + chunking
- `uploads/` — saved PDF uploads
- `outputs/summary.txt`, `outputs/flashcards.txt` — rewritten each run
- `.env.example`, `requirements.txt`

## Notes
- Model defaults: `command-r` (chat), `embed-english-v3.0` (embeddings)
- Change via `.env` if needed.
