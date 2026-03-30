"""Deconstructionist agent implementation."""

from typing import List

from langchain_core.documents import Document

from hegel_ai.agents.base import Agent, AgentRegistry
from hegel_ai.config import AgentConfig
from hegel_ai.logging_config import get_logger

logger = get_logger("agents.deconstructionist")


@AgentRegistry.register("deconstructionist")
class DeconstructionistAgent(Agent):
    """Deconstructionist philosophical agent."""

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
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content[:500]}..."
            for doc in retrieved
        ]) if retrieved else "[ERROR: No excerpts retrieved]"

        previous_args = self._get_previous_arguments(debate_history)
        uniqueness_instruction = ""
        if previous_args:
            uniqueness_instruction = f"""
**UNIQUENESS REQUIREMENT:**
You have previously argued: {'; '.join(previous_args[-2:])}
Do NOT repeat these points. Add NEW insights or counter-arguments.
"""

        return f"""{self.config.system.format(concepts=self.config.concepts)}

**MANDATORY RESPONSE FORMAT:**

1. FIRST SENTENCE MUST BE: "As [EXACT_FILENAME] states: '[DIRECT_QUOTE]'"
   - Use an ACTUAL quote from the excerpts below
   - Include the EXACT filename with .txt extension
   - Quote at least 10 words verbatim

2. Then deconstruct using: {self.config.concepts}

3. Show how the text undermines its own claims

{uniqueness_instruction}

**CONSTRAINTS:**
- Word limit: 150-200 words
- Do not repeat arguments from debate history
- Use authentic phrases like: {self.config.example_phrases}

**RETRIEVED EXCERPTS (YOU MUST CITE ONE):**
{grounding_text}

**DEBATE HISTORY:**
{debate_history[-2000:]}

**YOUR RESPONSE (start with citation NOW):**"""

    def _get_previous_arguments(self, debate_history: str) -> List[str]:
        previous = []
        for line in debate_history.split("\n"):
            if line.startswith(f"{self.config.name}:"):
                previous.append(line.replace(f"{self.config.name}:", "").strip())
        return previous
