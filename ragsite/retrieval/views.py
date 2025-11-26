# retrieval/views.py
import os
import sys
import json
import uuid
import pickle
import traceback
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .forms import UploadFileForm

# --- Locate rag package and use absolute imports ---
# expected layout: <repo>/ragsite (BASE_DIR) and <repo>/ragsite/rag
RAG_PACKAGE_PATH = os.path.normpath(os.path.join(settings.BASE_DIR, "rag"))
# fallback: repo root /rag
if not os.path.isdir(RAG_PACKAGE_PATH):
    alt = os.path.normpath(os.path.join(settings.BASE_DIR, "..", "rag"))
    if os.path.isdir(alt):
        RAG_PACKAGE_PATH = alt

# Ensure parent dir is on sys.path so "import rag.*" works
parent_dir = os.path.dirname(RAG_PACKAGE_PATH)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now do absolute imports from rag
try:
    from rag.embedder import Embedder
    from rag.llm_client import get_answer_from_documents
    from rag.utils import chunk_text
except Exception:
    # show import traceback in server logs for debugging; endpoints return friendly JSON error
    traceback.print_exc()
    Embedder = None
    get_answer_from_documents = None
    chunk_text = None

# --- Try to import PDF helpers (they may be added to rag/utils.py) ---
try:
    from rag.utils import extract_text_from_pdf, sanitize_text_for_metadata
except Exception:
    # fallback implementations (safe) if rag.utils lacks them
    def extract_text_from_pdf(path: str) -> str:
        # best-effort: return empty so caller falls back to file read
        print(f"[warning] extract_text_from_pdf not available for {path}")
        return ""

    def sanitize_text_for_metadata(t: str, max_len: int = 1000) -> str:
        if not isinstance(t, str):
            return ""
        s = t.replace("\x00", " ")
        s = " ".join(s.split())
        return s[:max_len]

# --- faiss / numpy ---
import faiss
import numpy as np

# Files to persist (store inside rag package folder)
FAISS_INDEX_PATH = os.path.join(RAG_PACKAGE_PATH, "faiss_index.bin")
METADATA_PATH = os.path.join(RAG_PACKAGE_PATH, "metadata.pkl")

# Embedding model name (can be changed)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------- Helpers -------------------- #

def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as fh:
            return pickle.load(fh)
    return []

def save_metadata(metadata_list):
    with open(METADATA_PATH, "wb") as fh:
        pickle.dump(metadata_list, fh)

def is_index_idmap(index):
    try:
        return "IndexIDMap" in type(index).__name__
    except Exception:
        return False

def convert_index_to_idmap(index):
    """
    Safely return an IndexIDMap that preserves vectors if present.
    - If index already IDMap -> return it
    - If index not IDMap and empty -> wrap and return IndexIDMap(index)
    - If index not IDMap and non-empty -> reconstruct vectors and create new IDMap
    """
    if is_index_idmap(index):
        return index

    try:
        dim = int(index.d)
        ntotal = int(index.ntotal)
    except Exception:
        raise RuntimeError("Unable to read faiss index dimensions/ntotal.")

    if ntotal == 0:
        return faiss.IndexIDMap(index)

    # Reconstruct vectors and build new IDMap
    flat = faiss.IndexFlatIP(dim)
    idmap = faiss.IndexIDMap(flat)

    # Reconstruct all vectors into numpy array
    vecs = np.zeros((ntotal, dim), dtype="float32")
    buf = np.zeros(dim, dtype="float32")
    for i in range(ntotal):
        index.reconstruct(i, buf)   # fills buf
        vecs[i, :] = buf

    # Normalise if expected (we typically store normalized vectors)
    try:
        faiss.normalize_L2(vecs)
    except Exception:
        pass

    ids = np.arange(ntotal, dtype="int64")
    idmap.add_with_ids(vecs, ids)
    return idmap

def build_or_load_faiss_index(dim=None):
    """
    Load index file if exists and ensure it's an IndexIDMap.
    If no file, create new IndexIDMap(IndexFlatIP(dim)).
    """
    if os.path.exists(FAISS_INDEX_PATH):
        idx = faiss.read_index(FAISS_INDEX_PATH)
        idx = convert_index_to_idmap(idx)
        return idx
    else:
        if dim is None:
            raise RuntimeError("No FAISS index on disk and dim not provided.")
        flat = faiss.IndexFlatIP(dim)
        idx = faiss.IndexIDMap(flat)
        return idx

def save_faiss_index(index):
    faiss.write_index(index, FAISS_INDEX_PATH)

def get_next_vector_id(metadata_list):
    if not metadata_list:
        return 0
    return max(item.get("vector_id", -1) for item in metadata_list) + 1

def make_embedder_safe(model_name):
    """
    Try to initialize Embedder wrapper; raise RuntimeError with traceback if fails.
    """
    if Embedder is None:
        raise RuntimeError("Embedder module not available (import error).")
    try:
        e = Embedder(model_name=model_name)
        return e
    except Exception as ex:
        tb = traceback.format_exc()
        raise RuntimeError(f"Failed to initialize embedder: {ex}\n{tb}")

# -------------------- Django views -------------------- #

def index(request):
    return render(request, "retrieval/index.html")

@csrf_exempt
def upload_file(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    form = UploadFileForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({"error": "invalid form", "details": form.errors}, status=400)

    f = request.FILES["file"]
    title = form.cleaned_data["title"]
    uid = uuid.uuid4().hex
    filename = f"{uid}_{f.name}"
    save_dir = settings.MEDIA_ROOT
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "wb") as fh:
        for chunk in f.chunks():
            fh.write(chunk)

    return JsonResponse({"status": "uploaded", "file_path": save_path, "title": title})

@csrf_exempt
def index_document(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    # basic availability checks
    if chunk_text is None:
        return JsonResponse({"error": "chunk utility missing (import error)"}, status=500)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "invalid json body"}, status=400)

    file_path = payload.get("file_path")
    title = payload.get("title", None)

    if not file_path:
        return JsonResponse({"error": "file_path required"}, status=400)

    # normalize path relative to BASE_DIR if necessary
    if not os.path.isabs(file_path):
        file_path = os.path.join(settings.BASE_DIR, file_path)
    if not os.path.exists(file_path):
        return JsonResponse({"error": "file not found", "file_path": file_path}, status=404)

    # read file content: use PDF extractor for PDFs, otherwise read text
    try:
        raw = ""
        if file_path.lower().endswith(".pdf"):
            raw = extract_text_from_pdf(file_path) or ""
            # If PDF extraction returned nothing, fallback to raw read attempt (best-effort)
            if not raw:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                        raw = fh.read()
                except Exception:
                    raw = ""
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                raw = fh.read()
    except Exception as ex:
        return JsonResponse({"error": "failed to read file", "details": str(ex)}, status=500)

    # chunk text
    chunks = chunk_text(raw, chunk_size=1000, overlap=200)
    if not chunks:
        return JsonResponse({"status": "no_chunks"}, status=200)

    # load metadata and determine next id
    metadata = load_metadata()
    next_vid = get_next_vector_id(metadata)

    # init embedder safely
    try:
        embedder = make_embedder_safe(EMBED_MODEL)
    except RuntimeError as ex:
        return JsonResponse({"error": "embedder_init_failed", "details": str(ex)}, status=500)

    # prepare or load index
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            index = faiss.read_index(FAISS_INDEX_PATH)
            index = convert_index_to_idmap(index)
        else:
            # determine dim by encoding small sample
            sample = embedder.encode(["hello world"], batch_size=1)
            dim = int(sample.shape[1])
            index = build_or_load_faiss_index(dim=dim)
    except Exception as ex:
        tb = traceback.format_exc()
        return JsonResponse({"error": "faiss_index_init_failed", "details": str(ex), "trace": tb}, status=500)

    # encode in batches and add_with_ids
    batch_size = 32
    added = 0
    for i in range(0, len(chunks), batch_size):
        batch_texts = chunks[i:i+batch_size]
        try:
            emb = embedder.encode(batch_texts, batch_size=batch_size).astype("float32")
        except Exception as ex:
            tb = traceback.format_exc()
            return JsonResponse({"error": "embedding_failed", "details": str(ex), "trace": tb}, status=500)

        # normalize
        try:
            faiss.normalize_L2(emb)
        except Exception:
            pass

        n = emb.shape[0]
        ids = np.arange(next_vid, next_vid + n, dtype="int64")

        # ensure index is IDMap (convert if not)
        if not is_index_idmap(index):
            index = convert_index_to_idmap(index)

        try:
            index.add_with_ids(emb, ids)
        except Exception as ex:
            tb = traceback.format_exc()
            return JsonResponse({"error": "faiss_add_failed", "details": str(ex), "trace": tb}, status=500)

        # append metadata records (sanitize excerpts)
        for j, txt in enumerate(batch_texts):
            meta = {
                "vector_id": int(ids[j]),
                "doc_path": file_path,
                "title": title or os.path.basename(file_path),
                "chunk_index": i + j,
                "text": sanitize_text_for_metadata(txt)
            }
            metadata.append(meta)
            added += 1
        next_vid += n

    # persist
    try:
        save_faiss_index(index)
        save_metadata(metadata)
    except Exception as ex:
        tb = traceback.format_exc()
        return JsonResponse({"error": "persist_failed", "details": str(ex), "trace": tb}, status=500)

    return JsonResponse({"status": "indexed", "chunks_added": added})

@csrf_exempt
def query_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "invalid json"}, status=400)

    query = payload.get("query")
    top_k = int(payload.get("top_k", 3))

    if not query:
        return JsonResponse({"error": "query required"}, status=400)

    if Embedder is None:
        return JsonResponse({"error": "embedder module not available"}, status=500)

    if not os.path.exists(FAISS_INDEX_PATH):
        return JsonResponse({"error": "faiss index not found; index some docs first"}, status=400)

    # init embedder safely
    try:
        embedder = make_embedder_safe(EMBED_MODEL)
    except RuntimeError as ex:
        return JsonResponse({"error": "embedder_init_failed", "details": str(ex)}, status=500)

    try:
        q_emb = embedder.encode([query], batch_size=1).astype("float32")
    except Exception as ex:
        tb = traceback.format_exc()
        return JsonResponse({"error": "query_embedding_failed", "details": str(ex), "trace": tb}, status=500)

    try:
        faiss.normalize_L2(q_emb)
    except Exception:
        pass

    try:
        idx = faiss.read_index(FAISS_INDEX_PATH)
        idx = convert_index_to_idmap(idx)
    except Exception as ex:
        tb = traceback.format_exc()
        return JsonResponse({"error": "faiss_load_failed", "details": str(ex), "trace": tb}, status=500)

    D, I = idx.search(q_emb.reshape(1, -1), top_k)
    vector_ids = I[0].tolist()
    distances = D[0].tolist()

    metadata = load_metadata()

    # build mapping safely (skip broken metadata entries)
    meta_by_vid = {}
    for idx_i, m in enumerate(metadata):
        try:
            vid = int(m["vector_id"])
        except Exception:
            print(f"Warning: metadata entry {idx_i} missing/invalid 'vector_id'; skipping")
            continue
        meta_by_vid[vid] = m

    sources = []
    for pos, vid in enumerate(vector_ids):
        if vid == -1:
            continue
        m = meta_by_vid.get(int(vid))
        if m is None:
            continue
        sources.append({
            "vector_id": int(vid),
            "title": m.get("title"),
            "excerpt": m.get("text", "")[:400],
            "score": float(distances[pos])
        })

    # synthesize answer with LLM
    try:
        if get_answer_from_documents is None:
            answer = "LLM client not available"
        else:
            doc_texts = [s["excerpt"] for s in sources]

            raw_answer = get_answer_from_documents(question=query, documents=doc_texts)

            # normalize answer to plain string for UI
            try:
                if isinstance(raw_answer, str):
                    answer = raw_answer
                elif isinstance(raw_answer, dict):
                    # try common keys
                    answer = None
                    for key in ("text", "answer", "content", "output"):
                        if key in raw_answer and isinstance(raw_answer[key], str):
                            answer = raw_answer[key]
                            break
                    if answer is None:
                        # last resort: attempt to extract nested textual content
                        if "choices" in raw_answer and isinstance(raw_answer["choices"], list) and raw_answer["choices"]:
                            ch = raw_answer["choices"][0]
                            if isinstance(ch, dict) and "text" in ch:
                                answer = ch["text"]
                        if answer is None:
                            answer = str(raw_answer)
                elif isinstance(raw_answer, list):
                    answer = " ".join(str(x) for x in raw_answer)
                else:
                    # object-like responses
                    if hasattr(raw_answer, "text"):
                        answer = getattr(raw_answer, "text")
                    elif hasattr(raw_answer, "content"):
                        answer = str(getattr(raw_answer, "content"))
                    else:
                        answer = str(raw_answer)
            except Exception:
                answer = str(raw_answer)
    except Exception as ex:
        answer = f"LLM call failed: {str(ex)}"

    return JsonResponse({"answer": answer, "sources": sources})
