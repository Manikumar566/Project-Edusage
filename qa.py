import os, time, re, uuid
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
import cohere
from utils import extract_text_pages, chunk_text

load_dotenv()

COHERE_KEY = (os.getenv("COHERE_API_KEY") or "").strip()
CHAT_MODEL = (os.getenv("COHERE_CHAT_MODEL") or "command-r").strip()
EMB_MODEL  = (os.getenv("COHERE_EMBED_MODEL") or "embed-english-v3.0").strip()

if not COHERE_KEY:
    raise RuntimeError("COHERE_API_KEY missing. Put it in a .env file next to app.py")

co = cohere.Client(COHERE_KEY)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a) + 1e-9
    b_norm = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (a_norm * b_norm))

def _embed_documents(texts: List[str]) -> np.ndarray:
    for attempt in range(3):
        try:
            resp = co.embed(texts=texts, model=EMB_MODEL, input_type="search_document")
            return np.array(resp.embeddings, dtype=np.float32)
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate" in msg:
                time.sleep(1.2 * (attempt + 1))
                continue
            raise RuntimeError(f"Embedding (docs) error: {e}")
    raise RuntimeError("Embedding (docs) rate-limited. Try again.")

def _embed_query(text: str) -> np.ndarray:
    for attempt in range(3):
        try:
            resp = co.embed(texts=[text], model=EMB_MODEL, input_type="search_query")
            return np.array(resp.embeddings[0], dtype=np.float32)
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate" in msg:
                time.sleep(1.2 * (attempt + 1))
                continue
            raise RuntimeError(f"Embedding (query) error: {e}")
    raise RuntimeError("Embedding (query) rate-limited. Try again.")

def _chat(prompt: str, max_tokens: int = 350, temperature: float = 0.2) -> str:
    for attempt in range(3):
        try:
            resp = co.chat(model=CHAT_MODEL, message=prompt, temperature=temperature, max_tokens=max_tokens)
            return (resp.text or "").strip()
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate" in msg:
                time.sleep(1.2 * (attempt + 1))
                continue
            raise RuntimeError(f"Chat error: {e}")
    raise RuntimeError("Chat rate-limited. Try again in a few seconds.")

def _clean_para(text: str) -> str:
    txt = re.sub(r'\[source:.*?\]', '', text)
    txt = " ".join((txt or "").split())
    return txt.strip()

def _parse_flashcards(raw: str) -> list:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    cards = []
    q, a = None, None
    for line in lines:
        if line.lower().startswith("q:"):
            if q and a:
                cards.append({"question": q, "answer": a})
            q = line[2:].strip()
            a = None
        elif line.lower().startswith("a:"):
            a = line[2:].strip()
        else:
            if a is None and q is not None:
                q += " " + line
            elif a is not None:
                a += " " + line
    if q and a:
        cards.append({"question": q, "answer": a})
    if not cards:
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                cards.append({"question": lines[i], "answer": lines[i+1]})
    return cards[:8]

class QASystem:
    def __init__(self):
        self.chunks: List[Dict] = []
        self.outputs_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        open(os.path.join(self.outputs_dir, "summary.txt"), "w").close()
        open(os.path.join(self.outputs_dir, "flashcards.txt"), "w").close()

    def index_pdf(self, pdf_path: str):
        name = os.path.basename(pdf_path)
        pages = extract_text_pages(pdf_path)
        new_chunks = []
        for page_num, text in pages:
            for ch in chunk_text(text):
                new_chunks.append({"id": str(uuid.uuid4()), "text": ch, "source": name, "page": page_num})
        if not new_chunks:
            return
        vecs = _embed_documents([c["text"] for c in new_chunks])
        for c, v in zip(new_chunks, vecs):
            c["vec"] = v
        self.chunks.extend(new_chunks)

    def _retrieve(self, question: str, k: int = 6) -> List[Dict]:
        if not self.chunks:
            return []
        qv = _embed_query(question)
        scored = [(cosine(qv, c["vec"]), c) for c in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]

    def answer(self, question: str) -> str:
        if not self.chunks:
            return "No PDFs indexed yet. Upload in the sidebar."
        top = self._retrieve(question, k=6)
        ctx = "\n\n".join([f"[{c['source']} p.{c['page']}] {c['text']}" for c in top])
        prompt = (
            "You are EduSage, an academic assistant.\n"
            "Answer ONLY using the context from the user's PDF below.\n"
            "If the answer is not present, say you cannot find it in the PDF.\n"
            "Respond as a single concise paragraph. No bullet points, no lists.\n\n"
            "Question:\n" + question + "\n\nContext:\n" + ctx + "\n\nAnswer:"
        )
        out = _chat(prompt, max_tokens=400, temperature=0.2)
        return _clean_para(out)

    def summarize_all(self) -> str:
        if not self.chunks:
            return "No PDFs indexed yet. Upload in the sidebar."
        texts = [c["text"] for c in self.chunks[:60]]
        ctx = "\n\n".join(texts)
        prompt = (
            "Summarize the following academic content into 6-8 crisp bullet points using simple language. "
            "Do not include citations or references.\n\n" + ctx + "\n\nSummary:"
        )
        out = _chat(prompt, max_tokens=450, temperature=0.3)
        summary = out.strip()
        with open(os.path.join(self.outputs_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary)
        return summary

    def flashcards(self) -> list:
        if not self.chunks:
            return [{"question": "No data", "answer": "Upload PDFs first."}]
        texts = [c["text"] for c in self.chunks[:50]]
        ctx = "\n\n".join(texts)
        prompt = (
            "From the following academic content, create 6-8 study flashcards in the format:\n"
            "Q: <short question>\nA: <short answer>\n\n"
            "Keep them concise and exam-oriented.\n\n" + ctx + "\n\nFlashcards:"
        )
        out = _chat(prompt, max_tokens=500, temperature=0.3)
        cards = _parse_flashcards(out)
        
        with open(os.path.join(self.outputs_dir, "flashcards.txt"), "w", encoding="utf-8") as f:
            for i, c in enumerate(cards, 1):
                f.write(f"Q{i}: {c['question']}\nA{i}: {c['answer']}\n\n")
        return cards
