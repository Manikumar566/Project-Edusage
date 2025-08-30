import fitz

def extract_text_pages(pdf_path: str):
    """Return list of (page_num_starting_at_1, text)."""
    pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        try:
            pages.append((i, page.get_text()))
        except Exception:
            pages.append((i, ""))
    doc.close()
    return pages

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200):
    text = " ".join((text or "").split())
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+max_chars])
        i += max_chars - overlap
    return chunks
