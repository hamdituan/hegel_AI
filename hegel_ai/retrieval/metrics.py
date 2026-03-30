"""Retrieval quality metrics."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document

from hegel_ai.logging_config import get_logger

logger = get_logger("retrieval.metrics")


@dataclass
class RetrievalMetrics:
    """Retrieval metrics."""
    query: str
    total_results: int = 0
    filtered_results: int = 0
    avg_similarity_score: float = 0.0
    max_similarity_score: float = 0.0
    min_similarity_score: float = 0.0
    diversity_score: float = 1.0
    front_matter_filtered: int = 0
    relevance_threshold: float = 0.3

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "total_results": self.total_results,
            "filtered_results": self.filtered_results,
            "avg_similarity_score": round(self.avg_similarity_score, 4),
            "max_similarity_score": round(self.max_similarity_score, 4),
            "min_similarity_score": round(self.min_similarity_score, 4),
            "diversity_score": round(self.diversity_score, 4),
            "front_matter_filtered": self.front_matter_filtered,
            "relevance_threshold": self.relevance_threshold,
        }

    def __str__(self) -> str:
        return (
            f"RetrievalMetrics(query='{self.query[:50]}...', "
            f"results={self.filtered_results}/{self.total_results}, "
            f"avg_sim={self.avg_similarity_score:.3f}, "
            f"diversity={self.diversity_score:.3f})"
        )


FRONT_MATTER_PATTERNS = [
    "introduction",
    "preface",
    "translator",
    "editor's note",
    "table of contents",
    "acknowledgments",
    "publisher's note",
    "contents",
    "foreword",
]


def is_front_matter(text: str, patterns: Optional[List[str]] = None) -> bool:
    if patterns is None:
        patterns = FRONT_MATTER_PATTERNS

    text_lower = text.lower()
    return any(pattern in text_lower for pattern in patterns)


def compute_diversity(docs: List[Document]) -> float:
    if len(docs) < 2:
        return 1.0

    def get_word_set(text: str) -> set:
        words = set(text.lower().split())
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "it", "its", "this", "that", "these", "those", "i", "you", "he",
            "she", "we", "they", "what", "which", "who", "whom", "whose",
        }
        return words - stopwords

    word_sets = [get_word_set(doc.page_content) for doc in docs]

    total_similarity = 0.0
    count = 0

    for i, set1 in enumerate(word_sets):
        for set2 in word_sets[i + 1:]:
            if set1 and set2:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                if union > 0:
                    total_similarity += intersection / union
                    count += 1

    avg_similarity = total_similarity / count if count > 0 else 0.0
    return 1.0 - avg_similarity


def apply_mmr(
    results: List[Tuple[Document, float]],
    top_k: int,
    lambda_mult: float = 0.7,
) -> List[Tuple[Document, float]]:
    if len(results) <= top_k:
        return results

    if not results:
        return []

    selected = [results[0]]
    remaining = list(results[1:])

    while len(selected) < top_k and remaining:
        best_score = -float("inf")
        best_idx = 0

        for i, (doc, sim_score) in enumerate(remaining):
            max_sim_to_selected = max(
                _cosine_similarity_words(doc.page_content, sel_doc.page_content)
                for sel_doc, _ in selected
            )

            mmr_score = (
                lambda_mult * sim_score
                - (1 - lambda_mult) * max_sim_to_selected
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(remaining[best_idx])
        remaining.pop(best_idx)

    return selected


def _cosine_similarity_words(text1: str, text2: str) -> float:
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0
