"""Load FAISS index and metadata; provide a simple query entrypoint.

Usage: python retriever.py --query "your question" --top-k 5
"""
import argparse
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llm_client import get_answer_from_documents

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def load_index_and_meta(index_file, meta_file):
    import os
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"FAISS index file not found: {index_file}")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")
    index = faiss.read_index(index_file)
    with open(meta_file, 'rb') as fh:
        metadata = pickle.load(fh)
    return index, metadata


def search(query: str, index_file: str, meta_file: str, top_k: int = 5):
    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode(query, convert_to_numpy=True).astype('float32')
    # normalize for cosine
    faiss.normalize_L2(q_emb.reshape(1, -1))

    index, metadata = load_index_and_meta(index_file, meta_file)
    D, I = index.search(q_emb.reshape(1, -1), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        meta = metadata[idx]
        results.append({
            'doc_id': meta.get('doc_path'),
            'title': meta.get('title'),
            'excerpt': meta.get('text')[:400],
            'score': float(score)
        })
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--query', required=True)
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--index', default='faiss_index.bin', help='Path to FAISS index file')
    p.add_argument('--db', default='metadata.pkl', help='Path to metadata pickle file')
    args = p.parse_args()
    try:
        res = search(args.query, index_file=args.index, meta_file=args.db, top_k=args.top_k)
        import json
        print(json.dumps({'answer': 'See sources', 'confidence': 'medium', 'sources': res}, indent=2))
        print("\n\n")
        ans = get_answer_from_documents(args.query, json.dumps(res))
        print(f"Answer from LLM: {ans}")

    except FileNotFoundError as e:
        print(f"Error: {e}\nMake sure you specify the correct --index and --db file paths. If you used batch_indexer.py, the default metadata file is 'metadata.pkl'.")
