# RUNNING.md — Setup & Run Instructions

This document explains how to set up, configure, and run the NovaDocs-AI project locally. It focuses on practical steps: environment setup, running the web UI, performing batch indexing, and using the retriever CLI.

## Prerequisites

- Python 3.8 or higher
- Git
- (Optional) CUDA-enabled GPU if you want FAISS/GPU and faster embeddings

## 1 — Clone & Create Virtual Environment

```powershell
git clone https://github.com/Anujdwi/NovaDocs-AI.git
cd NovaDocs-AI

# Create a virtual environment (Windows PowerShell)
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux use `python3 -m venv .venv` and `source .venv/bin/activate`.

## 2 — Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- If `faiss-cpu` fails to build, install the prebuilt wheel (or try `faiss-gpu` on a machine with CUDA).
- `cohere` is required for LLM features; if you don't need LLM output you can skip it.

## 3 — Configure Environment Variables

Create a `.env` file in the project root (or set env vars in your shell):

```
COHERE_API_KEY=your_cohere_api_key_here
COHERE_MODEL=command-xlarge-nightly
DEBUG=True
SECRET_KEY=replace_with_a_django_secret
```

For PowerShell (temporary session):

```powershell
$env:COHERE_API_KEY = "your_cohere_api_key_here"
```

## 4 — Django: Run Server (Web UI)

Start the Django server from the `ragsite` folder:

```powershell
cd ragsite
python manage.py migrate
python manage.py runserver 8000
```

Open `http://localhost:8000` in a browser. Use the UI to upload files, index them, and ask queries.

## 5 — Batch Indexing (CLI)

Use `batch_indexer.py` to convert a folder of text files into a FAISS index and metadata pickle.

```powershell
cd ragsite/rag
python batch_indexer.py --archive "C:\path\to\archive" --index faiss_index.bin --db metadata.pkl --batch-size 64
```

Output files:
- `faiss_index.bin` — FAISS index (IndexIDMap) with vector IDs
- `metadata.pkl` — list of metadata dicts with `vector_id` fields

If you want to run on a smaller test set, point `--archive` to a folder with a few `.txt` files.

## 6 — Retriever CLI (quick queries)

Run quick queries against an existing index:

```powershell
cd ragsite/rag
python retriever.py --query "What is this project about?" --index faiss_index.bin --db metadata.pkl --top-k 5
```

The script prints sources and then calls the LLM (if configured) to produce an answer.

## 7 — Troubleshooting & Common Fixes

- `COHERE_API_KEY not set`: Export the key or create `.env` and restart shell.
- `faiss` installation issues: install `faiss-cpu` or use conda for complex setups.
- `ImportError: sentence_transformers`: `pip install sentence-transformers`.

## 8 — Testing Locally

1. Index a small set of files with `batch_indexer.py`.
2. Run `retriever.py` to confirm it returns sources and LLM answers.
3. Start the Django server and run the same query from the UI.

## 9 — Security & Best Practices

- Do not commit `.env` or any API keys. Add `.env` to `.gitignore`.
- Rotate API keys if accidentally exposed.

## 10 — Next Steps / Improvements

- Add Docker support for reproducible environments.
- Add unit tests for the embedder, indexer, and retriever.
- Add CI to run linting and tests.

---

If you want, I can also create a small `env.example` file and update `.gitignore` to ignore `.env` (I can do that next).

Last updated: November 26, 2025
