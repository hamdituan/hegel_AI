"""Self-Refinement prompting - agents critique and improve their arguments."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tqdm import tqdm

from hegel_ai.llm.ollama_client import get_llm_client
from hegel_ai.logging_config import get_logger

logger = get_logger("prompts.refinement")


@dataclass
class RefinementCritique:
    """Refinement critique."""
    strengths: List[str]
    weaknesses: List[str]
    logical_gaps: List[str]
    evidence_issues: List[str]
    suggestions: List[str]

    def to_prompt(self) -> str:
        lines = ["CRITIQUE SUMMARY:"]

        if self.strengths:
            lines.append("\nStrengths:")
            for s in self.strengths:
                lines.append(f"  + {s}")

        if self.weaknesses:
            lines.append("\nWeaknesses:")
            for w in self.weaknesses:
                lines.append(f"  - {w}")

        if self.logical_gaps:
            lines.append("\nLogical Gaps:")
            for g in self.logical_gaps:
                lines.append(f"  ! {g}")

        if self.evidence_issues:
            lines.append("\nEvidence Issues:")
            for e in self.evidence_issues:
                lines.append(f"  ? {e}")

        if self.suggestions:
            lines.append("\nSuggestions:")
            for s in self.suggestions:
                lines.append(f"  > {s}")

        return "\n".join(lines)


@dataclass
class RefinementResult:
    """Refinement result."""
    original: str
    critique: RefinementCritique
    refined: str
    improvements_made: List[str]
    num_passes: int


class SelfRefinement:
    """Self-refinement prompting."""

    def __init__(
        self,
        num_passes: int = 2,
        critique_temperature: float = 0.5,
        revision_temperature: float = 0.6,
    ):
        self.num_passes = num_passes
        self.critique_temperature = critique_temperature
        self.revision_temperature = revision_temperature
        self._client = get_llm_client()

    def critique(
        self,
        argument: str,
        passage: str,
        agent_concepts: str,
        citation_source: str,
        citation_quote: str,
    ) -> RefinementCritique:
        prompt = f"""You are critically reviewing your own philosophical argument.

**YOUR ARGUMENT:**
{argument}

**PASSAGE BEING ANALYZED:**
"{passage}"

**YOUR PHILOSOPHICAL LENS:**
{agent_concepts}

**YOUR CITATION:**
Source: {citation_source}
Quote: "{citation_quote}"

---
**CRITIQUE CHECKLIST:**

1. **Citation Quality**:
   - Is the quote actually from the cited source?
   - Is the quote relevant to your argument?
   - Is the quote long enough to be meaningful (10+ words)?

2. **Logical Coherence**:
   - Does your reasoning follow logically?
   - Are there any logical leaps or gaps?
   - Do your conclusions follow from your premises?

3. **Conceptual Depth**:
   - Are you using your philosophical concepts correctly?
   - Could you go deeper with any concept?
   - Are you missing any relevant concepts?

4. **Engagement with Passage**:
   - Are you directly addressing the passage?
   - Are you analyzing the passage or just making general claims?
   - Could you engage more specifically with the language?

5. **Originality**:
   - Is your argument insightful or obvious?
   - Are you repeating common interpretations?
   - What unique perspective can you add?

6. **Clarity and Precision**:
   - Is your writing clear and precise?
   - Are any sentences confusing or ambiguous?
   - Could you express any point more effectively?

---
Respond in this exact format:

STRENGTHS:
- [Strength 1]
- [Strength 2]

WEAKNESSES:
- [Weakness 1]
- [Weakness 2]

LOGICAL GAPS:
- [Gap 1 or "None identified"]

EVIDENCE ISSUES:
- [Issue 1 or "None identified"]

SUGGESTIONS:
- [Suggestion 1]
- [Suggestion 2]
"""

        try:
            response = self._client.generate_with_retry(
                prompt=prompt,
                temperature=self.critique_temperature,
                max_tokens=500,
            )
            return self._parse_critique(response)
        except Exception as e:
            logger.error(f"Critique generation failed: {e}")
            return RefinementCritique(
                strengths=["Argument addresses the passage"],
                weaknesses=["Could not generate detailed critique"],
                logical_gaps=[],
                evidence_issues=[],
                suggestions=["Proceed with revision based on available information"],
            )

    def _parse_critique(self, response: str) -> RefinementCritique:
        strengths = []
        weaknesses = []
        logical_gaps = []
        evidence_issues = []
        suggestions = []

        current_section = None

        for line in response.split("\n"):
            line = line.strip()

            if line.startswith("STRENGTHS:"):
                current_section = "strengths"
            elif line.startswith("WEAKNESSES:"):
                current_section = "weaknesses"
            elif line.startswith("LOGICAL GAPS:"):
                current_section = "logical_gaps"
            elif line.startswith("EVIDENCE ISSUES:"):
                current_section = "evidence_issues"
            elif line.startswith("SUGGESTIONS:"):
                current_section = "suggestions"
            elif line.startswith("-") and current_section:
                item = line[1:].strip()
                if item and item.lower() != "none identified":
                    if current_section == "strengths":
                        strengths.append(item)
                    elif current_section == "weaknesses":
                        weaknesses.append(item)
                    elif current_section == "logical_gaps":
                        logical_gaps.append(item)
                    elif current_section == "evidence_issues":
                        evidence_issues.append(item)
                    elif current_section == "suggestions":
                        suggestions.append(item)

        return RefinementCritique(
            strengths=strengths,
            weaknesses=weaknesses,
            logical_gaps=logical_gaps,
            evidence_issues=evidence_issues,
            suggestions=suggestions,
        )

    def revise(
        self,
        original: str,
        critique: RefinementCritique,
        passage: str,
        agent_concepts: str,
    ) -> Tuple[str, List[str]]:
        prompt = f"""You are revising your philosophical argument based on self-critique.

**ORIGINAL ARGUMENT:**
{original}

**YOUR CRITIQUE:**
{critique.to_prompt()}

**PASSAGE:**
"{passage}"

**PHILOSOPHICAL CONCEPTS:**
{agent_concepts}

---
**REVISION TASK:**

1. Address the weaknesses identified
2. Fill logical gaps where possible
3. Improve evidence usage
4. Enhance clarity and precision
5. Maintain your core argument while improving execution

**IMPORTANT:**
- Keep your citation intact (do not change the source or quote)
- Maintain your philosophical perspective
- Aim for 150-200 words
- Make concrete improvements, not just rewording

---
Respond in this exact format:

REVISED ARGUMENT:
[Your improved argument]

IMPROVEMENTS MADE:
- [Improvement 1]
- [Improvement 2]
- [Improvement 3]
"""

        try:
            response = self._client.generate_with_retry(
                prompt=prompt,
                temperature=self.revision_temperature,
                max_tokens=500,
            )
            return self._parse_revision(response)
        except Exception as e:
            logger.error(f"Revision failed: {e}")
            return original, ["Could not complete revision"]

    def _parse_revision(self, response: str) -> Tuple[str, List[str]]:
        revised = ""
        improvements = []

        current_section = None

        for line in response.split("\n"):
            line = line.strip()

            if line.startswith("REVISED ARGUMENT:"):
                current_section = "revised"
            elif line.startswith("IMPROVEMENTS MADE:"):
                current_section = "improvements"
            elif current_section == "revised" and line:
                revised += line + "\n"
            elif current_section == "improvements" and line.startswith("-"):
                improvements.append(line[1:].strip())

        if not improvements and revised:
            improvements = ["Refined argument based on self-critique"]

        return revised.strip(), improvements

    def refine(
        self,
        initial_argument: str,
        passage: str,
        agent_concepts: str,
        citation_source: str,
        citation_quote: str,
    ) -> RefinementResult:
        logger.info(f"Running self-refinement ({self.num_passes} passes)")

        current_argument = initial_argument
        all_improvements = []
        final_critique = None

        with tqdm(total=self.num_passes, desc="Self-Refinement", unit="pass") as pbar:
            for i in range(self.num_passes):
                pbar.set_description(f"Refinement pass {i + 1}/{self.num_passes}")

                critique = self.critique(
                    argument=current_argument,
                    passage=passage,
                    agent_concepts=agent_concepts,
                    citation_source=citation_source,
                    citation_quote=citation_quote,
                )
                final_critique = critique

                revised, improvements = self.revise(
                    original=current_argument,
                    critique=critique,
                    passage=passage,
                    agent_concepts=agent_concepts,
                )

                if revised and len(revised) > 50:
                    current_argument = revised
                    all_improvements.extend(improvements)
                    logger.debug(f"Pass {i + 1}: {len(improvements)} improvements")

                pbar.update(1)

        return RefinementResult(
            original=initial_argument,
            critique=final_critique or RefinementCritique([], [], [], [], []),
            refined=current_argument,
            improvements_made=all_improvements,
            num_passes=self.num_passes,
        )
