import os
from typing import List


def list_text_files(folder: str) -> List[str]:
    """Recursively list .txt files in folder."""
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.lower().endswith('.txt'):
                files.append(os.path.join(root, fn))
    return sorted(files)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple text chunking by characters with overlap.

    This is intentionally simple: it keeps words intact by splitting on whitespace
    boundaries after taking the desired window.
    """
    if chunk_size <= 0:
        return [text]
    text = text.replace('\r\n', '\n')
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > chunk_size:
            chunks.append(' '.join(cur))
            # start new chunk with overlap words
            if overlap > 0:
                # approximate overlap in words
                ol_words = max(1, int(overlap / (max(1, chunk_size) / max(1, len(cur))))) if cur else []
                # fallback: keep last N words up to overlap chars
                # simple approach: keep last 50 words (safe small overlap)
                cur = cur[-50:]
            else:
                cur = []
            cur_len = sum(len(x) for x in cur) + len(cur) - 1 if cur else 0
        cur.append(w)
        cur_len += len(w) + 1
    if cur:
        chunks.append(' '.join(cur))
    return chunks
