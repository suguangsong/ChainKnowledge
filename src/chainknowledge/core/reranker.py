from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from langchain_core.documents import Document


def _tokenize(text: str) -> List[str]:
    normalized = (text or "").lower()
    return re.findall(r"[\w\u4e00-\u9fff]+", normalized)


def _jaccard(query_tokens: set[str], text_tokens: set[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = query_tokens.intersection(text_tokens)
    return len(overlap) / len(query_tokens.union(text_tokens))


@dataclass(frozen=True)
class SimpleHybridReranker:
    """Simple hybrid reranker: vector score + lexical score."""

    alpha: float = 0.7

    def __post_init__(self) -> None:
        if self.alpha < 0.0:
            object.__setattr__(self, "alpha", 0.0)
        elif self.alpha > 1.0:
            object.__setattr__(self, "alpha", 1.0)

    def rerank(
        self,
        query: str,
        documents_with_scores: Sequence[tuple[Document, float]],
        top_k: int,
    ) -> List[Document]:
        if not documents_with_scores or top_k <= 0:
            return []

        query_tokens = set(_tokenize(query))
        ranked: list[tuple[float, Document]] = []
        for document, raw_score in documents_with_scores:
            vector_score = 1.0 / (1.0 + max(0.0, float(raw_score)))
            lexical_score = 0.0
            if query_tokens:
                lexical_score = _jaccard(query_tokens, set(_tokenize(document.page_content or "")))
            final_score = self.alpha * vector_score + (1.0 - self.alpha) * lexical_score
            ranked.append((final_score, document))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in ranked[:top_k]]


def create_reranker(enabled: bool, alpha: float | None) -> SimpleHybridReranker | None:
    if not enabled:
        return None
    return SimpleHybridReranker(alpha=alpha if alpha is not None else 0.7)
