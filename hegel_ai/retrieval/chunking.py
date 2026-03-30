"""Improved document chunking with multi-level semantic splitting."""

import logging
from typing import List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from hegel_ai.logging_config import get_logger

logger = get_logger("retrieval.chunking")


def semantic_chunk_document(
    doc: Document,
    embedding_model,
    percentile: int = 75,
    min_chunk_sentences: int = 2,
    max_chunk_sentences: int = 15,
    target_chunk_sentences: int = 8,
    use_overlap: bool = True,
    overlap_sentences: int = 2,
) -> List[Document]:
    """
    Improved semantic chunking with multi-level splitting and overlap.
    
    Args:
        doc: Document to chunk
        embedding_model: Embedding model for sentence embeddings
        percentile: Split threshold (lower = more chunks). 75 = split at 25% most dissimilar
        min_chunk_sentences: Minimum sentences per chunk
        max_chunk_sentences: Maximum sentences per chunk
        target_chunk_sentences: Target sentences per chunk (for secondary splitting)
        use_overlap: Whether to add overlap between chunks
        overlap_sentences: Number of sentences to overlap
    """
    text = doc.page_content

    if not text.strip():
        logger.warning(f"Empty document: {doc.metadata.get('source', 'unknown')}")
        return []

    sentences = sent_tokenize(text)

    if len(sentences) <= min_chunk_sentences:
        return [Document(
            page_content=text,
            metadata=doc.metadata.copy()
        )]

    if len(sentences) > 5000:
        logger.warning(
            f"Document {doc.metadata.get('source', 'unknown')} has {len(sentences)} sentences. "
            f"Truncating to 5000."
        )
        sentences = sentences[:5000]

    embeddings = _get_sentence_embeddings(sentences, doc, embedding_model)
    if not embeddings:
        return [Document(page_content=text, metadata=doc.metadata.copy())]

    similarities = _compute_similarities(embeddings)

    if len(similarities) == 0:
        return [Document(page_content=text, metadata=doc.metadata.copy())]

    split_indices = _find_split_points_multi_level(
        similarities=similarities,
        total_sentences=len(sentences),
        percentile=percentile,
        min_chunk_sentences=min_chunk_sentences,
        max_chunk_sentences=max_chunk_sentences,
        target_chunk_sentences=target_chunk_sentences,
    )

    chunks: List[Document] = []
    chunk_index = 0

    for i, split_idx in enumerate(split_indices):
        start = split_indices[i - 1] if i > 0 else 0
        end = split_idx

        chunk_sentences = sentences[start:end]

        if len(chunk_sentences) >= min_chunk_sentences:
            chunk_overlap = ""
            if use_overlap and i > 0 and start >= overlap_sentences:
                overlap_start = start - overlap_sentences
                chunk_overlap = " ".join(sentences[overlap_start:start]) + " "

            chunk_text = chunk_overlap + " ".join(chunk_sentences)

            meta = doc.metadata.copy()
            meta["chunk_index"] = chunk_index
            meta["sentence_count"] = len(chunk_sentences)
            meta["char_count"] = len(chunk_text)
            meta["has_overlap"] = use_overlap and i > 0
            meta["start_sentence"] = start
            meta["end_sentence"] = end

            chunks.append(Document(page_content=chunk_text, metadata=meta))
            chunk_index += 1

    if not chunks:
        chunks = [Document(page_content=text, metadata=doc.metadata.copy())]

    logger.debug(f"Created {len(chunks)} chunks from {doc.metadata.get('source', 'unknown')}")
    return chunks


def _get_sentence_embeddings(
    sentences: List[str],
    doc: Document,
    embedding_model,
    batch_size: int = 32,
) -> List[List[float]]:
    """Get embeddings for all sentences with error handling."""
    embeddings: List[List[float]] = []

    try:
        with tqdm(
            total=len(sentences),
            desc=f"  Embedding: {doc.metadata.get('source', 'unknown')[:50]}",
            unit="sent",
            leave=False,
            ncols=80,
        ) as pbar:
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                batch_emb = embedding_model.embed_documents(batch)
                embeddings.extend(batch_emb)
                pbar.update(len(batch))
    except Exception as e:
        logger.error(f"Embedding failed for {doc.metadata.get('source')}: {e}")
        return []

    return embeddings


def _compute_similarities(embeddings: List[List[float]]) -> np.ndarray:
    """Compute cosine similarities between consecutive sentences."""
    embeddings_array = np.array(embeddings)

    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    norm_embeddings = embeddings_array / norms

    similarities = np.sum(norm_embeddings[:-1] * norm_embeddings[1:], axis=1)
    return similarities


def _find_split_points_multi_level(
    similarities: np.ndarray,
    total_sentences: int,
    percentile: int,
    min_chunk_sentences: int,
    max_chunk_sentences: int,
    target_chunk_sentences: int,
) -> List[int]:
    """
    Find split points using multi-level approach.
    
    Level 1: Split at major semantic boundaries (low percentile)
    Level 2: Split large chunks at moderate boundaries
    Level 3: Ensure no chunk exceeds max size
    """
    split_indices = set()

    threshold_coarse = np.percentile(similarities, percentile)
    threshold_fine = np.percentile(similarities, percentile + 10)

    for i, sim in enumerate(similarities):
        sentence_idx = i + 1

        if sim < threshold_coarse:
            if sentence_idx >= min_chunk_sentences and sentence_idx <= total_sentences - min_chunk_sentences:
                split_indices.add(sentence_idx)

        elif sim < threshold_fine:
            if sentence_idx >= target_chunk_sentences and sentence_idx <= total_sentences - min_chunk_sentences:
                split_indices.add(sentence_idx)

    sorted_splits = sorted(split_indices)
    final_splits = []
    prev_split = 0

    for split in sorted_splits:
        if split - prev_split >= min_chunk_sentences:
            final_splits.append(split)
            prev_split = split

    if total_sentences - prev_split > max_chunk_sentences:
        remaining = total_sentences - prev_split
        num_additional_splits = remaining // max_chunk_sentences

        for i in range(1, num_additional_splits + 1):
            additional_split = prev_split + (i * max_chunk_sentences)
            if additional_split < total_sentences - min_chunk_sentences:
                final_splits.append(additional_split)

    final_splits.append(total_sentences)

    return final_splits


def _create_chunk_doc(
    sentences: List[str],
    metadata: dict,
    index: int,
    overlap_text: str = "",
    start_sentence: int = 0,
    end_sentence: int = 0,
) -> Document:
    """Create a chunk document with metadata."""
    chunk_text = overlap_text + " ".join(sentences)
    meta = metadata.copy()
    meta["chunk_index"] = index
    meta["sentence_count"] = len(sentences)
    meta["char_count"] = len(chunk_text)
    meta["has_overlap"] = bool(overlap_text)
    meta["start_sentence"] = start_sentence
    meta["end_sentence"] = end_sentence

    return Document(page_content=chunk_text, metadata=meta)


def chunk_documents(
    docs: List[Document],
    embedding_model,
    percentile: int = 75,
    min_chunk_sentences: int = 2,
    max_chunk_sentences: int = 15,
    target_chunk_sentences: int = 8,
    use_overlap: bool = True,
    overlap_sentences: int = 2,
) -> List[Document]:
    """
    Chunk documents using improved semantic chunking.
    
    Args:
        docs: List of documents to chunk
        embedding_model: Embedding model
        percentile: Split threshold (lower = more chunks)
        min_chunk_sentences: Minimum sentences per chunk
        max_chunk_sentences: Maximum sentences per chunk
        target_chunk_sentences: Target sentences per chunk
        use_overlap: Whether to add overlap between chunks
        overlap_sentences: Number of sentences to overlap
    """
    all_chunks: List[Document] = []

    for doc in tqdm(docs, desc="Semantic chunking", unit="doc", ncols=80):
        try:
            doc_chunks = semantic_chunk_document(
                doc=doc,
                embedding_model=embedding_model,
                percentile=percentile,
                min_chunk_sentences=min_chunk_sentences,
                max_chunk_sentences=max_chunk_sentences,
                target_chunk_sentences=target_chunk_sentences,
                use_overlap=use_overlap,
                overlap_sentences=overlap_sentences,
            )
            all_chunks.extend(doc_chunks)
        except Exception as e:
            logger.error(
                f"Failed to chunk {doc.metadata.get('source', 'unknown')}: {e}"
            )
            all_chunks.append(doc)

    logger.info(f"Created {len(all_chunks):,} total chunks from {len(docs)} documents")
    return all_chunks


def estimate_chunk_count(
    docs: List[Document],
    target_chunk_sentences: int = 8,
) -> int:
    """Estimate expected chunk count before processing."""
    total_sentences = 0
    for doc in docs:
        sentences = sent_tokenize(doc.page_content)
        total_sentences += len(sentences)

    estimated_chunks = total_sentences // target_chunk_sentences
    return max(estimated_chunks, len(docs))
