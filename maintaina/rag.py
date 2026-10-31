"""Retrieval over a LOCAL, user-supplied knowledge base (maintenance manuals / guides).

Design notes
- Retrieval only: the app returns quoted passages with source + page. It does not generate or invent instructions.
- Backend: TF-IDF vectors + cosine similarity (light enough for free hosting). An embedding model + FAISS can be
  swapped in behind the same `search` interface (see README).
- Retrieved text is shown as quoted evidence only. It is never executed or treated as instructions to the system.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .config import KNOWLEDGE_DIR

SUPPORTED = {".txt", ".md", ".pdf"}

# Maps unusual sensor signals to search words. These are only SEARCH TERMS, not maintenance advice.
SIGNAL_TERMS = {
    "tool_wear": "tool wear inspection replacement",
    "torque": "torque overload load inspection",
    "rotational_speed": "rotational speed vibration bearing",
    "air_temperature": "temperature cooling ventilation",
    "process_temperature": "temperature overheating cooling heat dissipation",
}


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None


def _read(path: Path) -> list[tuple[str, int | None]]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return [((p.extract_text() or ""), i + 1) for i, p in enumerate(PdfReader(str(path)).pages)]
    return [(path.read_text(encoding="utf-8", errors="ignore"), None)]


def chunk_text(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    """Word-based chunks with overlap so a sentence cut at a boundary still appears whole in a neighbour."""
    words = re.sub(r"\s+", " ", text).split()
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step) if words[i:i + size]]


class KnowledgeBase:
    def __init__(self, size: int = 120, overlap: int = 30, min_score: float = 0.12):
        self.size, self.overlap, self.min_score = size, overlap, min_score
        self.chunks: list[Chunk] = []
        self._vec = self._mat = None

    def ingest_dir(self, directory: Path = KNOWLEDGE_DIR) -> int:
        self.chunks = []
        for p in sorted(Path(directory).rglob("*")):
            if p.suffix.lower() in SUPPORTED and p.name.lower() != "readme.md":
                for text, page in _read(p):
                    self.chunks += [Chunk(c, p.name, page) for c in chunk_text(text, self.size, self.overlap)]
        if self.chunks:
            self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self._mat = self._vec.fit_transform([c.text for c in self.chunks])
        else:
            self._vec = self._mat = None
        return len(self.chunks)

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.chunks:
            return []
        sims = linear_kernel(self._vec.transform([query]), self._mat).ravel()
        out = []
        for i in np.argsort(-sims)[:k]:
            if sims[i] >= self.min_score:  # below this: treat as "no evidence"
                c = self.chunks[i]
                out.append({"text": c.text, "source": c.source, "page": c.page, "score": round(float(sims[i]), 3)})
        return out


def build_query(unusual_inputs: list[dict], alert: bool, anomaly: bool) -> str:
    terms = [SIGNAL_TERMS[u["feature"]] for u in unusual_inputs if u["feature"] in SIGNAL_TERMS]
    base = "failure risk inspection maintenance procedure" if (alert or anomaly) else "routine inspection"
    return " ".join([base] + terms[:3])


def guidance(kb: KnowledgeBase, query: str, k: int = 3) -> dict:
    hits = kb.search(query, k)
    if not kb.chunks:
        msg = "No maintenance documents are loaded. Add manuals to the knowledge/ folder and re-index."
    elif not hits:
        msg = "No sufficiently relevant passage found in the loaded documents. No guidance is given."
    else:
        msg = "Passages below are quoted from your documents; verify them against the original manual."
    return {"query": query, "passages": hits, "message": msg}


def evaluate_retrieval(kb: KnowledgeBase, qa: list[tuple[str, str]], k: int = 3) -> float:
    """hit@k: share of questions whose expected source file appears in the top-k results."""
    hits = sum(any(r["source"] == src for r in kb.search(q, k)) for q, src in qa)
    return hits / max(len(qa), 1)
