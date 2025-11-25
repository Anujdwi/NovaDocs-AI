# Backend: FAISS indexer & retriever

This folder contains simple Python scripts to:

- convert text files into embeddings using `sentence-transformers`
- store embeddings in a FAISS index
- retrieve the top-k most similar document chunks for a query

Files:

- `utils.py` — helpers to list files and chunk text
- `indexer.py` — build and persist the FAISS index and metadata
- `retriever.py` — load index and metadata and run a query
- `requirements.txt` — python dependencies

# DocExtract Backend Usage Instructions

## 1. Create and Activate a Virtual Environment

Open PowerShell in your `backend` directory and run:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Install Required Packages

With the venv activated, install dependencies:

```powershell
pip install -r requirements.txt
```

## 3. Run the Batch Indexer to Create the Knowledge Base

Replace the archive path with your actual folder:

```powershell
python batch_indexer.py --archive "d:\DocExtract\archive"
```

This will create `faiss_index.bin` and `metadata.pkl` in your backend folder.

## 4. Run the Retriever to Query the Knowledge Base

Ask a question (replace with your actual question):

```powershell
python retriever.py --query "What is this project about?"
```

You can also specify custom index or metadata file paths if needed:

```powershell
python retriever.py --index faiss_index.bin --db metadata.pkl --query "Your question here"
```

You’re ready to use your knowledge base!
