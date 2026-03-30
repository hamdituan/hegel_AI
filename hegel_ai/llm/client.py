"""LLM client abstraction."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from hegel_ai.logging_config import get_logger

logger = get_logger("llm.client")


class LLMError(Exception):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        pass

    @abstractmethod
    def validate_response(self, response: str) -> bool:
        pass

    def generate_with_retry(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    wait_time = (2 ** attempt) + (attempt * 0.5)
                    logger.warning(
                        f"Retry {attempt}/{self.max_retries} for {self.model} "
                        f"in {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)

                response = self.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

                if not self.validate_response(response):
                    raise LLMError(f"Invalid response: {response[:100]}...")

                return response

            except LLMRateLimitError as e:
                last_error = e
                logger.error(f"Rate limit exceeded (attempt {attempt + 1}): {e}")

            except LLMTimeoutError as e:
                last_error = e
                logger.error(f"Timeout (attempt {attempt + 1}): {e}")

            except LLMError as e:
                last_error = e
                logger.error(f"LLM error (attempt {attempt + 1}): {e}")

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")

        raise LLMError(
            f"Generation failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    def truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        estimated_tokens = len(prompt) // 4

        if estimated_tokens <= max_tokens:
            return prompt

        ratio = max_tokens / estimated_tokens
        truncated_len = int(len(prompt) * ratio)

        return prompt[:truncated_len] + "\n...[truncated for length]"
