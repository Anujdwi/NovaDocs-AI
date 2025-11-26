# NovaDocs-AI — Project Overview

NovaDocs-AI is a Retrieval-Augmented Generation (RAG) system that combines semantic search with LLM-powered answers. It lets you upload documents, build a FAISS vector index of document chunks, and ask natural-language questions that are answered using retrieved context and a language model (Cohere).

Key highlights:
- Document upload, chunking, and embedding
- FAISS vector index with sequential vector IDs
- Cohere Chat integration for generating user-friendly answers
- Web UI for upload, indexing, and query; lightweight CLI tools for batch processing

For detailed setup and run instructions, see `RUNNING.md` in the repository root.

Sections in this repo:
- `ragsite/` — Django project (web UI, API endpoints)
- `ragsite/rag/` — RAG utilities (embedder, batch indexer, retriever, llm client)
- `archive/` — sample documents used for batch indexing

If you want to contribute, open an issue or submit a pull request on GitHub.
