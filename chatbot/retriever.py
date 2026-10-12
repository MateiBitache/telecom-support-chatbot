"""Retrieval over a Markdown knowledge base using TF-IDF and cosine similarity."""

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion


@dataclass
class Chunk:
    """One question/answer section of the knowledge base."""

    source: str  # file name, e.g. "billing.md"
    title: str  # the "## ..." heading
    text: str  # the answer under the heading

    @property
    def full_text(self) -> str:
        return f"{self.title}\n{self.text}"


def load_chunks(kb_dir: str | Path) -> list[Chunk]:
    """Split every .md file into chunks, one per '## ' heading."""
    chunks: list[Chunk] = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        title, lines = None, []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if title:
                    chunks.append(Chunk(path.name, title, "\n".join(lines).strip()))
                title, lines = line[3:].strip(), []
            elif title is not None:
                lines.append(line)
        if title:
            chunks.append(Chunk(path.name, title, "\n".join(lines).strip()))
    if not chunks:
        raise ValueError(f"No knowledge base sections found in {kb_dir}")
    return chunks


class Retriever:
    """Finds the knowledge base sections most similar to a question."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # Headings are repeated so the question wording weighs more than the answer text.
        documents = [f"{c.title} {c.title} {c.text}" for c in chunks]
        # Word n-grams capture exact terms; character n-grams handle word forms
        # ("stole" / "stolen") and small typos.
        self.vectorizer = FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])
        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks with their similarity score (0 to 1)."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in best]
