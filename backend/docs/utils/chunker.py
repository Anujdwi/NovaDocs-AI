import uuid

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Simple sliding-window chunker.
    Returns list of dicts: [{'chunk_id':..., 'text':..., 'meta': {...}}]
    """
    if not text:
        return []
    
    text = text.replace("\r\n", "\n")
    tokens = text.split()
    chunks = []
    i = 0
    n = len(tokens)
    while i < n:
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_text = " ".join(chunk_tokens)
        chunk_id = str(uuid.uuid4())
        meta = {"start_token": i, "end_token": min(i+chunk_size, n)}
        chunks.append({"chunk_id": chunk_id, "text": chunk_text, "meta": meta})
        i += chunk_size - overlap
    return chunks
    