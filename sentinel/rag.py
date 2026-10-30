"""Retrieval over each lender's policy documents.

Three strategies, selectable per deployment and compared by the retrieval eval: dense (embeddings,
catch meaning: "can I pay it off early?"), BM25 (exact terms: "origination fee", "Grade E") and
hybrid (weighted reciprocal rank fusion of both). The tenant filter is applied before anything is
scored, so one lender's documents can never reach another lender's applicants.
"""
import hashlib
import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
INDEX_PATH = KNOWLEDGE_DIR / "index.json"
EMBEDDING_MODEL = "text-embedding-3-small"
# Retrieval strategy is chosen by the retrieval eval (evals/retrieval_cases.jsonl), not by habit:
# on this corpus dense-only beats every BM25 fusion weight. See docs/decisions/003-retrieval.md
BM25_WEIGHT = 0.3
RetrievalMode = Literal["hybrid", "dense", "bm25"]


@dataclass
class Chunk:
    id: str            # e.g. "aster/03-rates-fees-repayment#fees"
    tenant_id: str
    doc_title: str
    section: str
    text: str          # title and section are prepended so the chunk makes sense on its own


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_chunks(root: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    """One chunk per '## ' section: sections are short and self-contained, so no overlap is needed."""
    chunks = []
    for path in sorted(root.glob("*/*.md")):
        tenant_id, doc = path.parent.name, path.stem
        title, *sections = re.split(r"^## ", path.read_text(), flags=re.M)
        doc_title = title.strip().removeprefix("# ").strip()
        for section in sections:
            heading, _, body = section.partition("\n")
            chunks.append(Chunk(
                id=f"{tenant_id}/{doc}#{_slug(heading)}", tenant_id=tenant_id, doc_title=doc_title,
                section=heading.strip(), text=f"{doc_title} > {heading.strip()}\n{body.strip()}",
            ))
    return chunks


def corpus_hash(chunks: list[Chunk]) -> str:
    return hashlib.sha256(json.dumps([asdict(c) for c in chunks]).encode()).hexdigest()[:16]


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    name = EMBEDDING_MODEL

    def __init__(self, api_key: str | None) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, timeout=10, max_retries=2)

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self.name, input=texts)
        return np.array([d.embedding for d in response.data], dtype=np.float32)


class HashingEmbedder:
    """Offline stand-in for tests: hashed bag of words. Similar wording gives similar vectors."""
    name = "hashing-512"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 512), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _tokens(text):
                out[i, int(hashlib.md5(token.encode()).hexdigest(), 16) % 512] += 1
        return out


_STOP = set("a an the of to and or in on for is are be can i my you your we our it if with what how do does".split())


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.tfs = [Counter(_tokens(d)) for d in docs]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lengths) / max(len(docs), 1)
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tfs))
        for term in _tokens(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                if f := tf.get(term):
                    norm = 1 - self.b + self.b * self.lengths[i] / self.avg_len
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out


class Retriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder,
                 mode: RetrievalMode = "dense") -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.mode = mode
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / np.where(norms == 0, 1, norms)
        self._by_tenant: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            self._by_tenant.setdefault(c.tenant_id, []).append(i)
        self._bm25 = {t: BM25([chunks[i].text for i in idx]) for t, idx in self._by_tenant.items()}
        self._embed_query = lru_cache(maxsize=1024)(self._embed_one)

    def _embed_one(self, query: str) -> tuple[float, ...]:
        vector = self.embedder.embed([query])[0]
        return tuple(vector / (np.linalg.norm(vector) or 1))

    def search(self, query: str, tenant_id: str, k: int = 4, rrf_k: int = 60, bm25_weight: float = BM25_WEIGHT,
               mode: RetrievalMode | None = None) -> list[SearchResult]:
        mode = mode or self.mode
        indices = self._by_tenant.get(tenant_id, [])  # tenant filter first, always
        if not indices or not query.strip():
            return []
        rankings: list[tuple[np.ndarray, float]] = []
        if mode in ("hybrid", "dense"):
            vector = np.array(self._embed_query(query.strip().lower()))
            rankings.append((self.embeddings[indices] @ vector, 1.0))
        if mode in ("hybrid", "bm25"):
            rankings.append((self._bm25[tenant_id].scores(query), bm25_weight if mode == "hybrid" else 1.0))
        fused = np.zeros(len(indices))
        for scores, weight in rankings:
            for rank, pos in enumerate(np.argsort(-scores)):
                fused[pos] += weight / (rrf_k + rank + 1)
        top = np.argsort(-fused)[:k]
        return [SearchResult(self.chunks[indices[p]], round(float(fused[p]), 4)) for p in top]


def build_index(embedder: Embedder, path: Path = INDEX_PATH) -> None:
    """Offline step: embed every chunk once. Run `python -m sentinel.rag` after editing documents."""
    chunks = load_chunks()
    vectors = embedder.embed([c.text for c in chunks])
    path.write_text(json.dumps({
        "embedding_model": embedder.name, "corpus_hash": corpus_hash(chunks),
        "chunks": [c.id for c in chunks], "embeddings": np.round(vectors, 5).tolist(),
    }))
    logger.info("Built index", extra={"fields": {"chunks": len(chunks), "path": str(path)}})


def load_retriever(embedder: Embedder, path: Path = INDEX_PATH, mode: RetrievalMode = "dense") -> Retriever:
    chunks = load_chunks()
    if embedder.name == EMBEDDING_MODEL and path.exists():
        index = json.loads(path.read_text())
        if index["corpus_hash"] == corpus_hash(chunks) and index["embedding_model"] == embedder.name:
            return Retriever(chunks, np.array(index["embeddings"], dtype=np.float32), embedder, mode)
        logger.warning("Policy index is stale; re-embedding documents (run `python -m sentinel.rag`)")
    return Retriever(chunks, embedder.embed([c.text for c in chunks]), embedder, mode)


if __name__ == "__main__":
    from .config import load_settings
    logging.basicConfig(level="INFO")
    build_index(OpenAIEmbedder(load_settings().openai_api_key))
    print(f"Wrote {INDEX_PATH}")
