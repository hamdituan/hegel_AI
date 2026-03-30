"""Vector database creation script."""

import logging
import shutil
import time

from hegel_ai.config import get_config
from hegel_ai.logging_config import setup_logging
from hegel_ai.retrieval.chunking import chunk_documents, estimate_chunk_count
from hegel_ai.retrieval.embeddings import get_embeddings
from hegel_ai.retrieval.vector_store import create_vector_store
from langchain_core.documents import Document
from pathlib import Path


def load_documents(data_dir: Path):
    """Load all .txt documents from data directory."""
    docs = []

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    txt_files = list(data_dir.rglob("*.txt"))

    if not txt_files:
        raise ValueError(f"No .txt files found in {data_dir}")

    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                logging.warning(f"Empty file: {file_path}")
                continue

            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "folder": file_path.parent.name,
                    "full_path": str(file_path),
                },
            )
            docs.append(doc)
            logging.info(f"Loaded: {file_path.name} ({len(content):,} chars)")

        except Exception as e:
            logging.error(f"Failed to read {file_path}: {e}")

    logging.info(f"Loaded {len(docs)} documents total")
    return docs


def main():
    setup_logging()

    config = get_config()

    if config.vector_db_dir.exists():
        db_file = config.vector_db_dir / "chroma.sqlite3"
        if db_file.exists():
            logging.warning("Vector database already exists.")
            response = input("Rebuild? (y/n): ").strip().lower()
            if response != "y":
                return

            shutil.rmtree(config.vector_db_dir)
            logging.info("Removed existing vector database")

    docs = load_documents(config.data_dir)
    if not docs:
        logging.error("No documents loaded. Aborting.")
        return

    total_chars = sum(len(doc.page_content) for doc in docs)
    logging.info(f"Total content: {total_chars:,} characters ({total_chars / 1e6:.2f} MB)")

    estimated_chunks = estimate_chunk_count(docs, target_chunk_sentences=8)
    logging.info(f"Estimated chunks: ~{estimated_chunks:,} (with target 8 sentences/chunk)")

    logging.info("\nLoading embedding model...")
    embeddings = get_embeddings()
    logging.info("Embedding model loaded")

    logging.info("\nChunking documents with improved semantic chunking...")
    logging.info(f"  Percentile: 75 (split at 25% most dissimilar)")
    logging.info(f"  Min sentences: 2")
    logging.info(f"  Max sentences: 15")
    logging.info(f"  Target sentences: 8")
    logging.info(f"  Overlap: 2 sentences")

    chunk_start = time.time()
    chunks = chunk_documents(
        docs=docs,
        embedding_model=embeddings,
        percentile=75,
        min_chunk_sentences=2,
        max_chunk_sentences=15,
        target_chunk_sentences=8,
        use_overlap=True,
        overlap_sentences=2,
    )
    chunk_time = time.time() - chunk_start

    if not chunks:
        logging.error("No chunks created. Aborting.")
        return

    logging.info(f"\nChunking complete: {len(chunks):,} chunks in {chunk_time:.1f}s")
    logging.info(f"Expansion ratio: {len(chunks) / len(docs):.1f}x")

    avg_sentences = sum(len(c.page_content.split('.')) for c in chunks[:100]) / min(100, len(chunks))
    logging.info(f"Average sentences per chunk: ~{avg_sentences:.1f}")

    logging.info("\nCreating vector store...")
    store_start = time.time()

    create_vector_store(
        chunks=chunks,
        persist_dir=config.vector_db_dir,
        batch_size=config.chunk_batch_size,
    )

    store_time = time.time() - store_start
    total_time = chunk_time + store_time

    logging.info(f"\n{'='*60}")
    logging.info("VECTOR DATABASE CREATION COMPLETE")
    logging.info(f"{'='*60}")
    logging.info(f"  Documents: {len(docs)}")
    logging.info(f"  Chunks: {len(chunks):,}")
    logging.info(f"  Chunking time: {chunk_time:.1f}s")
    logging.info(f"  Store creation time: {store_time:.1f}s")
    logging.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logging.info(f"  Location: {config.vector_db_dir}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
