"""Embedding model management."""

import logging
from typing import Optional, List, Union

from langchain_huggingface import HuggingFaceEmbeddings

from hegel_ai.logging_config import get_logger

logger = get_logger("retrieval.embeddings")


class EmbeddingManager:
    """Embedding model management."""

    _instance: Optional["EmbeddingManager"] = None
    _embeddings: Optional[HuggingFaceEmbeddings] = None

    def __new__(cls, *args, **kwargs) -> "EmbeddingManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
    ):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._initialized = True

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": self.device},
                    encode_kwargs={"normalize_embeddings": self.normalize},
                )
                logger.info(f"Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.get_embeddings()
        return embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.get_embeddings()
        return embeddings.embed_query(text)

    def clear_cache(self) -> None:
        self._embeddings = None
        logger.debug("Embedding model cache cleared")


def get_embeddings(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    normalize: Optional[bool] = None,
) -> HuggingFaceEmbeddings:
    from hegel_ai.config import get_config

    config = get_config()

    manager = EmbeddingManager(
        model_name=model_name or config.embedding_model,
        device=device or config.embedding_device,
        normalize=normalize if normalize is not None else config.embedding_normalize,
    )

    return manager.get_embeddings()
