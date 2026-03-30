"""Dialectical prompting - thesis-antithesis-synthesis structure."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

from hegel_ai.llm.ollama_client import get_llm_client
from hegel_ai.logging_config import get_logger

logger = get_logger("prompts.dialectical")


class DialecticalStage(Enum):
    THESIS = "thesis"
    ANTITHESIS = "antithesis"
    SYNTHESIS = "synthesis"


@dataclass
class DialecticalPosition:
    """A position in the dialectical process."""
    stage: DialecticalStage
    content: str
    counter_points: List[str] = None
    resolution: str = ""

    def __post_init__(self):
        if self.counter_points is None:
            self.counter_points = []


@dataclass
class DialecticalPrompt:
    """Dialectical prompting strategy."""

    enable_synthesis: bool = True
    synthesis_temperature: float = 0.5

    def __post_init__(self):
        self._client = get_llm_client()

    def generate_thesis(
        self,
        passage: str,
        agent_concepts: str,
        retrieved_excerpts: str,
        citation_source: str,
        citation_quote: str,
    ) -> DialecticalPosition:
        prompt = f"""You are presenting your THESIS - your initial philosophical argument.

**PASSAGE TO ANALYZE:**
"{passage}"

**YOUR PHILOSOPHICAL LENS:**
{agent_concepts}

**YOUR CITATION (MUST USE):**
Source: {citation_source}
Quote: "{citation_quote}"

**RETRIEVED EVIDENCE:**
{retrieved_excerpts}

---
**THESIS REQUIREMENTS:**

1. Begin with your citation: "As [{citation_source}] states: '{citation_quote}'"
2. Present your interpretation of the passage
3. Apply your philosophical concepts
4. Make a clear, arguable claim

This is your initial position. It will be challenged by counter-arguments.

---
Write your THESIS (150-200 words):
"""

        try:
            content = self._client.generate_with_retry(
                prompt=prompt,
                temperature=0.6,
                max_tokens=400,
            )
            return DialecticalPosition(
                stage=DialecticalStage.THESIS,
                content=content,
            )
        except Exception as e:
            logger.error(f"Thesis generation failed: {e}")
            return DialecticalPosition(
                stage=DialecticalStage.THESIS,
                content=f"[Thesis generation failed: {e}]",
            )

    def generate_antithesis(
        self,
        thesis: DialecticalPosition,
        opposing_agent_name: str,
        opposing_concepts: str,
        passage: str,
        retrieved_excerpts: str,
        citation_source: str,
        citation_quote: str,
    ) -> DialecticalPosition:
        counter_points = self._extract_claims(thesis.content)

        prompt = f"""You are presenting your ANTITHESIS - a counter-argument to the thesis.

**THESIS YOU ARE CHALLENGING:**
{thesis.content}

**KEY CLAIMS TO CHALLENGE:**
{chr(10).join(f"- {cp}" for cp in counter_points[:3])}

**PASSAGE:**
"{passage}"

**YOUR PHILOSOPHICAL LENS:**
{opposing_concepts}

**YOUR CITATION (MUST USE):**
Source: {citation_source}
Quote: "{citation_quote}"

**RETRIEVED EVIDENCE:**
{retrieved_excerpts}

---
**ANTITHESIS REQUIREMENTS:**

1. Begin with your citation: "As [{citation_source}] states: '{citation_quote}'"
2. Directly challenge the thesis claims
3. Offer an alternative interpretation
4. Use your philosophical concepts to support your challenge

Be respectful but rigorous. Your goal is to reveal limitations in the thesis.

---
Write your ANTITHESIS (150-200 words):
"""

        try:
            content = self._client.generate_with_retry(
                prompt=prompt,
                temperature=0.6,
                max_tokens=400,
            )
            return DialecticalPosition(
                stage=DialecticalStage.ANTITHESIS,
                content=content,
                counter_points=counter_points,
            )
        except Exception as e:
            logger.error(f"Antithesis generation failed: {e}")
            return DialecticalPosition(
                stage=DialecticalStage.ANTITHESIS,
                content=f"[Antithesis generation failed: {e}]",
                counter_points=counter_points,
            )

    def generate_synthesis(
        self,
        thesis: DialecticalPosition,
        antithesis: DialecticalPosition,
        synthesizing_agent_name: str,
        agent_concepts: str,
        passage: str,
        retrieved_excerpts: str,
        citation_source: str,
        citation_quote: str,
    ) -> DialecticalPosition:
        prompt = f"""You are creating a SYNTHESIS - reconciling the thesis and antithesis.

**THESIS:**
{thesis.content}

**ANTITHESIS:**
{antithesis.content}

**PASSAGE:**
"{passage}"

**YOUR PHILOSOPHICAL LENS:**
{agent_concepts}

**YOUR CITATION (MUST USE):**
Source: {citation_source}
Quote: "{citation_quote}"

**RETRIEVED EVIDENCE:**
{retrieved_excerpts}

---
**SYNTHESIS REQUIREMENTS:**

1. Begin with your citation: "As [{citation_source}] states: '{citation_quote}'"
2. Acknowledge valid points from BOTH thesis and antithesis
3. Identify what each perspective misses
4. Propose a higher-level understanding that incorporates both
5. Show how the tension reveals deeper truth about the passage

The synthesis should transcend the opposition, not just compromise.

---
Write your SYNTHESIS (150-200 words):
"""

        try:
            content = self._client.generate_with_retry(
                prompt=prompt,
                temperature=self.synthesis_temperature,
                max_tokens=450,
            )

            resolution = self._extract_resolution(thesis, antithesis, content)

            return DialecticalPosition(
                stage=DialecticalStage.SYNTHESIS,
                content=content,
                counter_points=thesis.counter_points + antithesis.counter_points,
                resolution=resolution,
            )
        except Exception as e:
            logger.error(f"Synthesis generation failed: {e}")
            return DialecticalPosition(
                stage=DialecticalStage.SYNTHESIS,
                content=f"[Synthesis generation failed: {e}]",
            )

    def _extract_claims(self, content: str) -> List[str]:
        claims = []

        indicators = [
            "argues that",
            "claims that",
            "suggests that",
            "implies that",
            "therefore",
            "thus",
            "this shows",
            "this means",
        ]

        sentences = content.replace("\n", " ").split(".")
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in indicators):
                if len(sentence) > 20 and len(sentence) < 200:
                    claims.append(sentence + ".")

        return claims[:5] if claims else ["The argument makes several interpretive claims"]

    def _extract_resolution(
        self,
        thesis: DialecticalPosition,
        antithesis: DialecticalPosition,
        synthesis_content: str,
    ) -> str:
        resolution_prompt = f"""Based on this synthesis, what is the resolution of the tension?

**THESIS CLAIM:** {thesis.counter_points[0] if thesis.counter_points else "Initial interpretation"}
**ANTITHESIS CHALLENGE:** {antithesis.counter_points[0] if antithesis.counter_points else "Counter-interpretation"}

**SYNTHESIS:**
{synthesis_content}

Extract in one sentence how the synthesis resolves or transcends the opposition:
"""

        try:
            resolution = self._client.generate_with_retry(
                prompt=resolution_prompt,
                temperature=0.3,
                max_tokens=100,
            )
            return resolution.strip()
        except Exception:
            return "The synthesis offers a reconciled perspective on the passage."

    def run_full_dialectic(
        self,
        passage: str,
        thesis_agent_concepts: str,
        antithesis_agent_concepts: str,
        synthesis_agent_concepts: str,
        thesis_excerpts: str,
        antithesis_excerpts: str,
        synthesis_excerpts: str,
        thesis_citation: tuple,
        antithesis_citation: tuple,
        synthesis_citation: tuple,
    ) -> Dict[str, DialecticalPosition]:
        logger.info("Running full dialectical cycle")

        thesis = self.generate_thesis(
            passage=passage,
            agent_concepts=thesis_agent_concepts,
            retrieved_excerpts=thesis_excerpts,
            citation_source=thesis_citation[0],
            citation_quote=thesis_citation[1],
        )
        logger.info("Thesis generated")

        antithesis = self.generate_antithesis(
            thesis=thesis,
            opposing_agent_name="Antithesis Agent",
            opposing_concepts=antithesis_agent_concepts,
            passage=passage,
            retrieved_excerpts=antithesis_excerpts,
            citation_source=antithesis_citation[0],
            citation_quote=antithesis_citation[1],
        )
        logger.info("Antithesis generated")

        synthesis = None
        if self.enable_synthesis:
            synthesis = self.generate_synthesis(
                thesis=thesis,
                antithesis=antithesis,
                synthesizing_agent_name="Synthesis Agent",
                agent_concepts=synthesis_agent_concepts,
                passage=passage,
                retrieved_excerpts=synthesis_excerpts,
                citation_source=synthesis_citation[0],
                citation_quote=synthesis_citation[1],
            )
            logger.info("Synthesis generated")

        return {
            "thesis": thesis,
            "antithesis": antithesis,
            "synthesis": synthesis,
        }
