"""Vector store management."""

import logging
import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from hegel_ai.logging_config import get_logger
from hegel_ai.retrieval.embeddings import get_embeddings
from hegel_ai.retrieval.metrics import (
    RetrievalMetrics,
    is_front_matter,
    apply_mmr,
    compute_diversity,
)

logger = get_logger("retrieval.vector_store")


def create_vector_store(
    chunks: List[Document],
    persist_dir: Path,
    batch_size: int = 100,
) -> Chroma:
    from tqdm import tqdm

    logger.info(f"Creating vector store with {len(chunks)} chunks")
    logger.info(f"Persist directory: {persist_dir}")

    start_time = time.time()
    embeddings = get_embeddings()
    load_time = time.time() - start_time
    logger.info(f"Embedding model loaded in {load_time:.1f}s")

    start_time = time.time()
    vectorstore: Optional[Chroma] = None

    with tqdm(total=len(chunks), desc="Adding chunks", unit="chunk") as pbar:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=str(persist_dir),
                    )
                else:
                    vectorstore.add_documents(batch)

                pbar.update(len(batch))

            except Exception as e:
                logger.error(f"Failed to add batch {i // batch_size}: {e}")
                raise

    elapsed = time.time() - start_time

    try:
        count = vectorstore._collection.count()
        logger.info(f"Vector store created with {count:,} chunks in {elapsed:.1f}s")
    except Exception as e:
        logger.warning(f"Could not verify chunk count: {e}")
        logger.info(f"Vector store created in {elapsed:.1f}s")

    logger.info(f"Store saved to {persist_dir}")
    return vectorstore


def load_vector_store(
    persist_dir: Optional[Path] = None,
    validate: bool = True,
) -> Optional[Chroma]:
    from hegel_ai.config import get_config

    config = get_config()
    vector_db_dir = persist_dir or config.vector_db_dir

    if not vector_db_dir.exists():
        logger.error(f"Vector store directory not found: {vector_db_dir}")
        return None

    db_file = vector_db_dir / "chroma.sqlite3"
    if not db_file.exists():
        logger.error(f"Chroma database file not found: {db_file}")
        return None

    if validate:
        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            conn.close()

            if result != "ok":
                logger.error(f"Vector store integrity check failed: {result}")
                return None

            logger.debug("Vector store integrity check passed")

        except sqlite3.Error as e:
            logger.error(f"Failed to validate vector store: {e}")
            return None

    try:
        embeddings = get_embeddings()

        vectorstore = Chroma(
            persist_directory=str(vector_db_dir),
            embedding_function=embeddings,
        )

        try:
            count = len(vectorstore.get(include=[]).get("ids", []))
        except Exception:
            try:
                count = vectorstore._collection.count()
            except Exception:
                count = 0

        if count == 0:
            logger.warning("Vector store is empty. Run create_vector_db.py first.")
            return None

        logger.info(f"Loaded vector store with {count:,} chunks from {vector_db_dir}")
        return vectorstore

    except ImportError as e:
        logger.error(
            f"Missing dependency: {e}. "
            f"Run: pip install langchain-chroma langchain-huggingface"
        )
        return None

    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return None


def retrieve_with_metrics(
    vectorstore: Chroma,
    query: str,
    top_k: int = 5,
    min_relevance_threshold: float = 0.3,
    enforce_diversity: bool = True,
    filter_front_matter: bool = True,
) -> Tuple[List[Document], Optional[RetrievalMetrics]]:
    if vectorstore is None:
        logger.error("Cannot retrieve: vectorstore is None")
        return [], None

    try:
        fetch_k = top_k * 3
        results_with_scores = vectorstore.similarity_search_with_score(query, k=fetch_k)

        if not results_with_scores:
            logger.warning(f"No results found for query: {query[:50]}...")
            metrics = RetrievalMetrics(
                query=query,
                total_results=0,
                filtered_results=0,
                relevance_threshold=min_relevance_threshold,
            )
            return [], metrics

        front_matter_count = 0
        filtered_results: List[Tuple[Document, float]] = []

        for doc, score in results_with_scores:
            similarity = 1.0 / (1.0 + score) if score >= 0 else 0.0

            if filter_front_matter and is_front_matter(doc.page_content):
                front_matter_count += 1
                continue

            if similarity >= min_relevance_threshold:
                filtered_results.append((doc, similarity))

        if enforce_diversity and len(filtered_results) > top_k:
            filtered_results = apply_mmr(filtered_results, top_k, lambda_mult=0.7)

        final_results = [doc for doc, _ in filtered_results[:top_k]]

        similarities = [sim for _, sim in filtered_results[:top_k]]

        metrics = RetrievalMetrics(
            query=query,
            total_results=len(results_with_scores),
            filtered_results=len(final_results),
            avg_similarity_score=sum(similarities) / len(similarities) if similarities else 0.0,
            max_similarity_score=max(similarities) if similarities else 0.0,
            min_similarity_score=min(similarities) if similarities else 0.0,
            diversity_score=compute_diversity(final_results),
            front_matter_filtered=front_matter_count,
            relevance_threshold=min_relevance_threshold,
        )

        logger.debug(f"Retrieval: {metrics}")
        return final_results, metrics

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return [], None


def get_vector_store_stats(vectorstore: Chroma) -> dict:
    try:
        count = len(vectorstore.get(include=[]).get("ids", []))
    except Exception:
        try:
            count = vectorstore._collection.count()
        except Exception:
            count = 0

    return {
        "document_count": count,
        "persist_directory": vectorstore._persist_directory,
    }
