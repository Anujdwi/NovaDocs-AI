import os
from typing import List
import pdfplumber
from PyPDF2 import PdfReader

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


def extract_text_from_pdf(path: str) -> str:
    """
    Extract text from a PDF file path.
    Tries pdfplumber first (better), then PyPDF2 as fallback.
    Returns empty string on failure.
    """
    try:
        # prefer pdfplumber if installed
        try:
            with pdfplumber.open(path) as pdf:
                pages = []
                for p in pdf.pages:
                    try:
                        txt = p.extract_text()
                        if txt:
                            pages.append(txt)
                    except Exception:
                        continue
                return "\n\n".join(pages).strip()
        except Exception:
            # fallback to PyPDF2
            reader = PdfReader(path)
            pages = []
            for p in reader.pages:
                try:
                    txt = p.extract_text() or ""
                    pages.append(txt)
                except Exception:
                    pages.append("")
            return "\n\n".join(pages).strip()
    except Exception as e:
        # log for debugging and return empty
        print(f"[extract_text_from_pdf] failed for {path}: {e}")
        return ""

def sanitize_text_for_metadata(t: str, max_len: int = 1000) -> str:
    """
    Remove nulls, collapse whitespace, and truncate for storing in metadata excerpts.
    """
    if not isinstance(t, str):
        return ""
    s = t.replace("\x00", " ")
    # collapse all whitespace to single spaces and strip
    s = " ".join(s.split())
    if max_len and len(s) > max_len:
        return s[:max_len]
    return s