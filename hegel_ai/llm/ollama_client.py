"""Ollama LLM client implementation."""

import logging
from typing import Optional, Any, Dict

import ollama
from ollama import ResponseError, RequestError

from hegel_ai.llm.client import LLMClient, LLMError, LLMTimeoutError, LLMRateLimitError
from hegel_ai.logging_config import get_logger

logger = get_logger("llm.ollama")


class OllamaClient(LLMClient):
    """Ollama LLM client."""

    def __init__(
        self,
        model: str = "gemma3:1b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        max_retries: int = 3,
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        self._model_verified = False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        try:
            options: Dict[str, Any] = {
                "temperature": temperature,
                "num_predict": max_tokens or 512,
                "top_p": 0.9,
            }

            options.update(kwargs.get("options", {}))

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options=options,
            )

            content = response["message"]["content"].strip()

            if not content:
                raise LLMError("Empty response from Ollama")

            logger.debug(f"Generated {len(content)} characters from {self.model}")
            return content

        except ResponseError as e:
            if e.status_code == 429:
                raise LLMRateLimitError(f"Rate limited by Ollama: {e}")
            elif e.status_code in (503, 504):
                raise LLMTimeoutError(f"Ollama service unavailable: {e}")
            else:
                raise LLMError(f"Ollama API error ({e.status_code}): {e}")

        except RequestError as e:
            raise LLMTimeoutError(f"Connection to Ollama failed: {e}")

        except KeyError as e:
            raise LLMError(f"Malformed Ollama response: {e}")

        except Exception as e:
            raise LLMError(f"Unexpected Ollama error: {e}")

    def validate_response(self, response: str) -> bool:
        if not response:
            logger.warning("Validation failed: empty response")
            return False

        if len(response) < 10:
            logger.warning(f"Validation failed: response too short ({len(response)} chars)")
            return False

        error_patterns = [
            "error:",
            "i cannot",
            "i'm unable",
            "i apologize",
            "as an ai",
        ]

        response_lower = response.lower()
        for pattern in error_patterns:
            if pattern in response_lower and pattern in response_lower[:100]:
                logger.warning(f"Validation failed: contains error pattern '{pattern}'")
                return False

        return True

    def is_model_available(self) -> bool:
        try:
            models = ollama.list()
            model_names = [m["name"] for m in models.get("models", [])]

            if self.model in model_names:
                return True

            model_base = self.model.split(":")[0]
            if any(m.startswith(model_base) for m in model_names):
                return True

            logger.warning(f"Model '{self.model}' not found. Available: {model_names}")
            return False

        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False

    def pull_model(self) -> bool:
        if self.is_model_available():
            return True

        logger.info(f"Pulling model: {self.model}")
        try:
            ollama.pull(self.model)
            logger.info(f"Model {self.model} pulled successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False


_client: Optional[OllamaClient] = None


def get_llm_client(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> OllamaClient:
    from hegel_ai.config import get_config

    config = get_config()

    global _client

    if _client is None:
        _client = OllamaClient(
            model=model or config.ollama_model,
            base_url=base_url or config.ollama_base_url,
            timeout=timeout or config.llm_timeout,
            max_retries=max_retries or config.llm_max_retries,
        )

    return _client


def generate_text(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    max_retries: Optional[int] = None,
) -> str:
    client = get_llm_client(model=model, max_retries=max_retries)
    return client.generate_with_retry(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
