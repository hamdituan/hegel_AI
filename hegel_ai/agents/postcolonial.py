"""Postcolonial agent implementation."""

from typing import List

from langchain_core.documents import Document

from hegel_ai.agents.base import Agent, AgentRegistry
from hegel_ai.config import AgentConfig
from hegel_ai.logging_config import get_logger

logger = get_logger("agents.postcolonial")


@AgentRegistry.register("postcolonial")
class PostcolonialAgent(Agent):
    """Postcolonial philosophical agent."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)

    def build_prompt(
        self,
        passage: str,
        retrieved: List[Document],
        debate_history: str,
        round_num: int,
    ) -> str:
        grounding_text = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content[:500]}"
            for doc in retrieved
        ]) if retrieved else "[ERROR: No excerpts retrieved]"

        dialectical_instruction = ""
        if self._dialectical:
            stage = self._dialectical.get_stage(round_num, 0)
            stage_instruction = self._dialectical.get_stage_instruction(stage, self.config.name)
            dialectical_instruction = f"\nDIALECTICAL STAGE ({stage}): {stage_instruction}\n"

        return f"""{self.config.system.format(concepts=self.config.concepts)}

{dialectical_instruction}

RETRIEVED EXCERPTS (YOU MUST CITE ONE AT THE START):
{grounding_text}

DEBATE HISTORY:
{debate_history[-1500:]}

PASSAGE TO ANALYZE: "{passage}"

YOUR RESPONSE (MUST START WITH CITATION):"""
