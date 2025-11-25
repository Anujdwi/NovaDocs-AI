"""Batch indexer that reads archive files, chunks them, batches embeddings using Embedder,
 and builds a FAISS index while persisting metadata to SQLite.

Usage example:
  python batch_indexer.py --archive "d:\\DocExtract\\archive" --db "metadata.db" --index "faiss_index.bin"
"""
import os
import argparse
import pickle
from tqdm import tqdm

import numpy as np
import faiss

from utils import list_text_files, chunk_text
from embedder import Embedder
# NOTE: removed SQLite persistence. Metadata will be saved as a pickle file.


DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def batch_index(archive_dir: str, db_path: str = 'metadata.pkl', index_path: str = 'faiss_index.bin',
                model_name: str = DEFAULT_MODEL, chunk_size: int = 1000, overlap: int = 200, batch_size: int = 64):

    # metadata will be collected and written as a pickle at `db_path`
    files = list_text_files(archive_dir)
    embedder = Embedder(model_name=model_name)

    # We'll collect vectors in batches and add to FAISS incrementally.
    vectors_list = []
    meta_buffer = []
    metadata = []

    dim = None
    index = None

    for f in tqdm(files, desc='Files'):
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                text = fh.read()
        except Exception:
            continue
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            meta_buffer.append({'doc_path': f, 'title': os.path.basename(f), 'chunk_index': i, 'text': chunk})
            # when buffer reaches batch_size, embed and add to index
            if len(meta_buffer) >= batch_size:
                texts = [m['text'] for m in meta_buffer]
                vectors = embedder.encode(texts, batch_size=batch_size)
                # normalize for cosine similarity
                faiss.normalize_L2(vectors)

                if index is None:
                    dim = vectors.shape[1]
                    index = faiss.IndexFlatIP(dim)
                index.add(vectors)
                # collect metadata in the same order vectors are added
                metadata.extend(meta_buffer)
                meta_buffer = []

    # flush remaining
    if meta_buffer:
        texts = [m['text'] for m in meta_buffer]
        vectors = embedder.encode(texts, batch_size=max(1, len(texts)))
        faiss.normalize_L2(vectors)
        if index is None:
            dim = vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    metadata.extend(meta_buffer)

    if index is None:
        print('No data indexed.')
        return

    faiss.write_index(index, index_path)
    # persist metadata as pickle
    try:
        with open(db_path, 'wb') as fh:
            pickle.dump(metadata, fh)
        print(f'Index saved to {index_path}. Total vectors: {index.ntotal}. Metadata saved to {db_path}')
    except Exception as e:
        print(f'Index saved to {index_path}. Total vectors: {index.ntotal}. Failed to write metadata: {e}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True)
    p.add_argument('--db', default='metadata.pkl', help='Path to metadata pickle file')
    p.add_argument('--index', default='faiss_index.bin')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--chunk-size', type=int, default=1000)
    p.add_argument('--overlap', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    batch_index(args.archive, db_path=args.db, index_path=args.index, model_name=args.model,
                chunk_size=args.chunk_size, overlap=args.overlap, batch_size=args.batch_size)
