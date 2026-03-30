"""Vector database creation script."""

import logging
import shutil

from hegel_ai.config import get_config
from hegel_ai.logging_config import setup_logging
from hegel_ai.retrieval.chunking import chunk_documents
from hegel_ai.retrieval.embeddings import get_embeddings
from hegel_ai.retrieval.vector_store import create_vector_store
from langchain_core.documents import Document
from pathlib import Path


def load_documents(data_dir: Path):
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

    docs = load_documents(config.data_dir)
    if not docs:
        logging.error("No documents loaded. Aborting.")
        return

    embeddings = get_embeddings()

    chunks = chunk_documents(docs, embeddings, percentile=90)
    if not chunks:
        logging.error("No chunks created. Aborting.")
        return

    create_vector_store(
        chunks=chunks,
        persist_dir=config.vector_db_dir,
        batch_size=config.chunk_batch_size,
    )


if __name__ == "__main__":
    main()
