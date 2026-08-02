"""Retrieval-augmented generation over the carrier/compliance corpus.

Design notes
------------
* Hybrid retrieval: dense embeddings (BGE-M3 in prod, MiniLM in dev) fused
  with BM42 sparse via Reciprocal Rank Fusion. The vector store is Qdrant,
  which natively serves both sides of the query.
* Reranking is pluggable: by default we rely on RRF fusion, which is free
  and fast. The optional cross-encoder path (sentence-transformers) can be
  enabled with the `brokeriq[rerank]` extra for higher precision.
* Every hit carries its source document + section so the report agent can
  cite facts instead of hallucinating them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Chunking: keep markdown section boundaries intact, split long sections on
# paragraph breaks so citations point at a specific, reviewable place.
_SECTION_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_MAX_CHUNK_CHARS = 1800


@dataclass
class Chunk:
    doc_id: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.doc_id} § {self.section}"


def split_markdown(text: str, doc_id: str) -> list[Chunk]:
    """Split a markdown corpus file into citation-ready chunks."""
    lines = text.splitlines()
    chunks: list[Chunk] = []
    current_section = "preamble"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(Chunk(doc_id=doc_id, section=current_section, text=body))
        buffer = []

    for line in lines:
        match = _SECTION_RE.match(line)
        if match:
            flush()
            current_section = match.group(2).strip()
            continue
        buffer.append(line)
        if len("\n".join(buffer)) >= _MAX_CHUNK_CHARS:
            flush()

    flush()
    return chunks


def load_corpus(corpus_dir: Path) -> list[Chunk]:
    """Load and chunk every *.md file under the corpus directory."""
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("skipping unreadable corpus file %s: %s", path.name, exc)
            continue
        chunks.extend(split_markdown(text, doc_id=path.stem))
    logger.info("loaded %d chunks from %s", len(chunks), corpus_dir)
    return chunks
