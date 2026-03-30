"""Document chunking."""

import logging
from typing import List, Optional

import numpy as np
from langchain_core.documents import Document
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from hegel_ai.logging_config import get_logger

logger = get_logger("retrieval.chunking")


def semantic_chunk_document(
    doc: Document,
    embedding_model,
    percentile: int = 90,
    min_chunk_sentences: int = 1,
    max_chunk_sentences: int = 20,
) -> List[Document]:
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

    if len(sentences) > max_chunk_sentences * 100:
        logger.warning(
            f"Document {doc.metadata.get('source', 'unknown')} has {len(sentences)} sentences. "
            f"Truncating to {max_chunk_sentences * 100}."
        )
        sentences = sentences[:max_chunk_sentences * 100]

    batch_size = 32
    embeddings: List[List[float]] = []

    try:
        with tqdm(
            total=len(sentences),
            desc=f"  Embedding: {doc.metadata.get('source', 'unknown')[:50]}",
            unit="sent",
            leave=False
        ) as pbar:
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                batch_emb = embedding_model.embed_documents(batch)
                embeddings.extend(batch_emb)
                pbar.update(len(batch))
    except Exception as e:
        logger.error(f"Embedding failed for {doc.metadata.get('source')}: {e}")
        return [Document(page_content=text, metadata=doc.metadata.copy())]

    embeddings_array = np.array(embeddings)

    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    norm_embeddings = embeddings_array / norms

    similarities = np.sum(norm_embeddings[:-1] * norm_embeddings[1:], axis=1)

    if len(similarities) == 0:
        return [Document(page_content=text, metadata=doc.metadata.copy())]

    threshold = np.percentile(similarities, percentile)
    split_indices = np.where(similarities < threshold)[0] + 1

    chunks: List[Document] = []
    start = 0

    for idx in split_indices:
        chunk_sentences = sentences[start:idx]

        if len(chunk_sentences) >= min_chunk_sentences:
            if len(chunk_sentences) > max_chunk_sentences:
                for i in range(0, len(chunk_sentences), max_chunk_sentences):
                    sub_chunk = chunk_sentences[i:i + max_chunk_sentences]
                    chunks.append(_create_chunk_doc(sub_chunk, doc.metadata, len(chunks)))
            else:
                chunks.append(_create_chunk_doc(chunk_sentences, doc.metadata, len(chunks)))

        start = idx

    if start < len(sentences):
        remaining = sentences[start:]
        if len(remaining) >= min_chunk_sentences:
            if len(remaining) > max_chunk_sentences:
                for i in range(0, len(remaining), max_chunk_sentences):
                    sub_chunk = remaining[i:i + max_chunk_sentences]
                    chunks.append(_create_chunk_doc(sub_chunk, doc.metadata, len(chunks)))
            else:
                chunks.append(_create_chunk_doc(remaining, doc.metadata, len(chunks)))

    if not chunks:
        chunks = [Document(page_content=text, metadata=doc.metadata.copy())]

    logger.debug(f"Created {len(chunks)} chunks from {doc.metadata.get('source', 'unknown')}")
    return chunks


def _create_chunk_doc(
    sentences: List[str],
    metadata: dict,
    index: int
) -> Document:
    chunk_text = " ".join(sentences)
    meta = metadata.copy()
    meta["chunk_index"] = index
    meta["sentence_count"] = len(sentences)
    meta["char_count"] = len(chunk_text)

    return Document(page_content=chunk_text, metadata=meta)


def chunk_documents(
    docs: List[Document],
    embedding_model,
    percentile: int = 90,
) -> List[Document]:
    all_chunks: List[Document] = []

    for doc in tqdm(docs, desc="Semantic chunking", unit="doc"):
        try:
            doc_chunks = semantic_chunk_document(doc, embedding_model, percentile)
            all_chunks.extend(doc_chunks)
        except Exception as e:
            logger.error(
                f"Failed to chunk {doc.metadata.get('source', 'unknown')}: {e}"
            )
            all_chunks.append(doc)

    logger.info(f"Created {len(all_chunks)} total chunks from {len(docs)} documents")
    return all_chunks
