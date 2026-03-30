"""Debate moderator."""

import logging
from typing import List, Optional

from hegel_ai.agents.base import AgentConfig
from hegel_ai.llm.ollama_client import get_llm_client
from hegel_ai.logging_config import get_logger

logger = get_logger("debate.moderator")


class Moderator:
    """Debate moderator."""

    def __init__(
        self,
        temperature: float = 0.4,
        max_tokens: int = 400,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = get_llm_client()

    def summarize(
        self,
        debate_history: str,
        agents: List[AgentConfig],
        max_history_chars: int = 4000,
    ) -> str:
        agent_names = ", ".join(agent.name for agent in agents)

        if len(debate_history) > max_history_chars:
            debate_history = debate_history[-max_history_chars:]

        prompt = f"""You are a neutral moderator in a philosophical debate.

Your task:
1. Summarize the key points of disagreement in 2-3 sentences
2. Pose ONE concise question that pushes the debate deeper
3. Specifically ask each critic to address a point from their unique theoretical lens
4. Encourage them to use concrete textual evidence

Debate Participants: {agent_names}

Debate History:
{debate_history}

Moderator Summary and Question:"""

        try:
            response = self._client.generate_with_retry(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            logger.debug(f"Moderator summary generated ({len(response)} chars)")
            return response

        except Exception as e:
            logger.error(f"Moderator summary failed: {e}")
            return self._fallback_summary(debate_history, agents)

    def _fallback_summary(
        self,
        debate_history: str,
        agents: List[AgentConfig],
    ) -> str:
        agent_names = ", ".join(agent.name for agent in agents)

        return (
            f"Debate Summary:\n"
            f"The discussion has involved perspectives from {agent_names}.\n\n"
            f"New Question:\n"
            f"How do your theoretical frameworks differently interpret the power dynamics "
            f"implicit in the passage? Please cite specific textual evidence."
        )


def moderator_summary(
    debate_history: str,
    agents: List[AgentConfig],
    temperature: float = 0.4,
) -> str:
    moderator = Moderator(temperature=temperature)
    return moderator.summarize(debate_history, agents)
