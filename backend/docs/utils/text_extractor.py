import os
from pathlib import Path


def extract_text_from_pdf(path):
    try:
        from PyPDF2 import PdfReader
    except Exception as e:
        raise ImportError("PyPDF2 is required for pdf extraction. Install 'pypdf' or 'pypdf2'.") from e
    
    text = []
    reader = PdfReader(path)
    for page in reader.pages:
        page_text = page.extract_text() or ''
        text.append(page_text)
    return '\n'.join(text)

def extract_text_from_text(file_path: str) -> str:
    """Extract text from plain .txt files."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"TXT extraction error: {e}")
        return ""


def extract_text_from_file(path):
    """Main function to extract text from various file formats."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in ['.txt', '.md']:
        return extract_text_from_text(path)
    if p.suffix.lower() in ['.pdf']:
        return extract_text_from_pdf(path)
    
    # Default: try to read as text
    return extract_text_from_text(path)