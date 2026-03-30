"""Tree of Thought prompting - agents explore multiple reasoning paths."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from tqdm import tqdm

from hegel_ai.llm.ollama_client import get_llm_client
from hegel_ai.logging_config import get_logger

logger = get_logger("prompts.tree_of_thought")


class BranchEvaluation(Enum):
    PROMISING = "promising"
    WEAK = "weak"
    CONTRADICTORY = "contradictory"
    REDUNDANT = "redundant"


@dataclass
class ThoughtBranch:
    """A reasoning branch."""
    id: int
    perspective: str
    reasoning: str = ""
    evidence: str = ""
    conclusion: str = ""
    evaluation: BranchEvaluation = BranchEvaluation.PROMISING
    score: float = 0.0

    def to_prompt(self) -> str:
        return f"""[Branch {self.id}: {self.perspective}]
Reasoning: {self.reasoning}
Evidence: {self.evidence}
Conclusion: {self.conclusion}
Evaluation: {self.evaluation.value} (score: {self.score:.2f})"""


@dataclass
class TreeOfThoughtResult:
    """ToT result."""
    branches: List[ThoughtBranch]
    selected_branch: ThoughtBranch
    synthesis: str = ""
    reasoning_trace: str = ""


class TreeOfThought:
    """Tree of Thought prompting."""

    def __init__(
        self,
        num_branches: int = 3,
        temperature_base: float = 0.7,
        evaluation_prompt: Optional[str] = None,
    ):
        self.num_branches = num_branches
        self.temperature_base = temperature_base
        self.evaluation_prompt = evaluation_prompt or self._default_evaluation_prompt()
        self._client = get_llm_client()

    def _default_evaluation_prompt(self) -> str:
        return """Evaluate this reasoning branch on:
1. **Coherence**: Does the reasoning follow logically?
2. **Evidence**: Is there sufficient textual support?
3. **Originality**: Does this offer a unique perspective?
4. **Relevance**: Does it address the passage directly?

Rate 0.0-1.0 and identify any weaknesses."""

    def generate_branches(
        self,
        passage: str,
        agent_concepts: str,
        retrieved_excerpts: str,
        debate_context: str = "",
    ) -> List[ThoughtBranch]:
        branches = []
        perspectives = self._generate_perspectives(agent_concepts, self.num_branches)

        with tqdm(total=self.num_branches, desc="Tree of Thought", unit="branch") as pbar:
            for i, perspective in enumerate(perspectives):
                pbar.set_description(f"Exploring: {perspective[:50]}...")
                branch = self._generate_single_branch(
                    branch_id=i + 1,
                    perspective=perspective,
                    passage=passage,
                    agent_concepts=agent_concepts,
                    retrieved_excerpts=retrieved_excerpts,
                    debate_context=debate_context,
                )
                branches.append(branch)
                pbar.update(1)

        return branches

    def _generate_perspectives(
        self,
        base_concepts: str,
        num_branches: int,
    ) -> List[str]:
        concept_list = [c.strip() for c in base_concepts.split(",")]

        perspectives = []

        if num_branches >= 3:
            perspectives.append("Textual Analysis: Close reading of language, syntax, and rhetorical structure")

            if concept_list:
                primary_concept = concept_list[0]
                perspectives.append(f"Conceptual Analysis: Examining how '{primary_concept}' operates in the passage")

            if len(concept_list) > 1:
                secondary_concept = concept_list[1]
                perspectives.append(f"Comparative Analysis: Tension between '{primary_concept}' and '{secondary_concept}'")
            else:
                perspectives.append("Comparative Analysis: Contrast with opposing philosophical positions")

            if num_branches > 3:
                perspectives.append("Implication Analysis: Logical consequences and implications of the argument")

        while len(perspectives) < num_branches:
            perspectives.append(f"Alternative Interpretation: Reconsidering the passage from a different angle")

        return perspectives[:num_branches]

    def _generate_single_branch(
        self,
        branch_id: int,
        perspective: str,
        passage: str,
        agent_concepts: str,
        retrieved_excerpts: str,
        debate_context: str,
    ) -> ThoughtBranch:
        prompt = f"""You are exploring ONE specific interpretive approach to analyze a philosophical passage.

**YOUR ASSIGNED PERSPECTIVE:** {perspective}

**PASSAGE TO ANALYZE:**
"{passage}"

**YOUR PHILOSOPHICAL CONCEPTS:**
{agent_concepts}

**RETRIEVED TEXTUAL EVIDENCE:**
{retrieved_excerpts}

**PREVIOUS DEBATE CONTEXT:**
{debate_context[-500:] if debate_context else "None yet"}

---
**TASK: Develop your reasoning step-by-step**

1. **Reasoning Chain** (3-4 steps):
   - Step 1: Initial observation from your perspective
   - Step 2: Connect to your philosophical concepts
   - Step 3: Integrate textual evidence
   - Step 4: Develop the implication

2. **Evidence Selection**: Quote the most relevant excerpt

3. **Preliminary Conclusion**: What does this branch suggest?

---
Respond in this exact format:

REASONING:
[Your step-by-step reasoning]

EVIDENCE:
[Your selected textual evidence with citation]

CONCLUSION:
[Your preliminary conclusion]
"""

        try:
            response = self._client.generate_with_retry(
                prompt=prompt,
                temperature=self.temperature_base,
                max_tokens=600,
            )

            reasoning, evidence, conclusion = self._parse_branch_response(response)

            return ThoughtBranch(
                id=branch_id,
                perspective=perspective,
                reasoning=reasoning,
                evidence=evidence,
                conclusion=conclusion,
            )

        except Exception as e:
            logger.error(f"Failed to generate branch {branch_id}: {e}")
            return ThoughtBranch(
                id=branch_id,
                perspective=perspective,
                reasoning="[Generation failed]",
                evidence="[No evidence]",
                conclusion="[No conclusion]",
                evaluation=BranchEvaluation.CONTRADICTORY,
                score=0.0,
            )

    def _parse_branch_response(self, response: str) -> tuple:
        reasoning = ""
        evidence = ""
        conclusion = ""

        sections = response.split("\n\n")
        for section in sections:
            if section.startswith("REASONING:"):
                reasoning = section.replace("REASONING:", "").strip()
            elif section.startswith("EVIDENCE:"):
                evidence = section.replace("EVIDENCE:", "").strip()
            elif section.startswith("CONCLUSION:"):
                conclusion = section.replace("CONCLUSION:", "").strip()

        if not reasoning:
            reasoning = response

        return reasoning, evidence, conclusion

    def evaluate_branches(
        self,
        branches: List[ThoughtBranch],
        passage: str,
    ) -> List[ThoughtBranch]:
        with tqdm(total=len(branches), desc="Evaluating branches", unit="branch") as pbar:
            for branch in branches:
                pbar.set_description(f"Evaluating branch {branch.id}")
                evaluation_prompt = f"""{self.evaluation_prompt}

**BRANCH TO EVALUATE:**
{branch.to_prompt()}

**ORIGINAL PASSAGE:**
"{passage}"

---
Provide your evaluation:
1. Strengths of this branch
2. Weaknesses or gaps
3. Score (0.0-1.0)
4. Evaluation: PROMISING, WEAK, CONTRADICTORY, or REDUNDANT

Respond in format:
STRENGTHS: [...]
WEAKNESSES: [...]
SCORE: [0.X]
EVALUATION: [status]
"""

                try:
                    response = self._client.generate_with_retry(
                        prompt=evaluation_prompt,
                        temperature=0.3,
                        max_tokens=300,
                    )

                    score, evaluation = self._parse_evaluation(response)
                    branch.score = score
                    branch.evaluation = evaluation
                    pbar.update(1)

                except Exception as e:
                    logger.warning(f"Evaluation failed for branch {branch.id}: {e}")
                    branch.score = 0.5
                    branch.evaluation = BranchEvaluation.PROMISING
                    pbar.update(1)

        return branches

    def _parse_evaluation(self, response: str) -> tuple:
        score = 0.5
        evaluation = BranchEvaluation.PROMISING

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score = float(line.replace("SCORE:", "").strip())
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    pass
            elif line.startswith("EVALUATION:"):
                eval_str = line.replace("EVALUATION:", "").strip().lower()
                if "weak" in eval_str:
                    evaluation = BranchEvaluation.WEAK
                elif "contradictory" in eval_str:
                    evaluation = BranchEvaluation.CONTRADICTORY
                elif "redundant" in eval_str:
                    evaluation = BranchEvaluation.REDUNDANT
                else:
                    evaluation = BranchEvaluation.PROMISING

        return score, evaluation

    def select_best_branch(self, branches: List[ThoughtBranch]) -> ThoughtBranch:
        viable = [
            b for b in branches
            if b.evaluation not in (BranchEvaluation.CONTRADICTORY, BranchEvaluation.REDUNDANT)
            and b.evidence
        ]

        if not viable:
            viable = branches

        best = max(viable, key=lambda b: b.score)
        logger.info(f"Selected branch {best.id} (score: {best.score:.2f})")

        return best

    def synthesize(
        self,
        branches: List[ThoughtBranch],
        selected: ThoughtBranch,
    ) -> str:
        synthesis_prompt = f"""You have explored multiple interpretive approaches to analyze a passage.

**SELECTED BRANCH (primary argument):**
{selected.to_prompt()}

**OTHER EXPLORED BRANCHES:**
{chr(10).join([b.to_prompt() for b in branches if b.id != selected.id])}

---
**TASK: Synthesize a coherent argument**

1. Use the SELECTED branch as your primary argument
2. Incorporate valuable insights from other branches where they strengthen your argument
3. Acknowledge alternative interpretations if they add nuance
4. Ensure your final argument is coherent and well-supported

Write your synthesized argument (2-3 paragraphs):
"""

        try:
            synthesis = self._client.generate_with_retry(
                prompt=synthesis_prompt,
                temperature=0.5,
                max_tokens=500,
            )
            return synthesis
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")
            return selected.conclusion

    def generate_reasoning_trace(self, branches: List[ThoughtBranch]) -> str:
        trace_lines = [
            "=" * 60,
            "TREE OF THOUGHT - REASONING TRACE",
            "=" * 60,
            "",
        ]

        for branch in branches:
            status = "SELECTED" if branch.evaluation == BranchEvaluation.PROMISING and branch.score >= 0.7 else ""
            trace_lines.append(f"Branch {branch.id}: {branch.perspective}")
            trace_lines.append(f"  Score: {branch.score:.2f} | Evaluation: {branch.evaluation.value} {status}")
            trace_lines.append("")

        trace_lines.append("=" * 60)

        return "\n".join(trace_lines)

    def run(
        self,
        passage: str,
        agent_concepts: str,
        retrieved_excerpts: str,
        debate_context: str = "",
    ) -> TreeOfThoughtResult:
        logger.info(f"Running Tree of Thought with {self.num_branches} branches")

        branches = self.generate_branches(
            passage=passage,
            agent_concepts=agent_concepts,
            retrieved_excerpts=retrieved_excerpts,
            debate_context=debate_context,
        )

        branches = self.evaluate_branches(branches, passage)

        selected = self.select_best_branch(branches)

        synthesis = self.synthesize(branches, selected)

        reasoning_trace = self.generate_reasoning_trace(branches)

        return TreeOfThoughtResult(
            branches=branches,
            selected_branch=selected,
            synthesis=synthesis,
            reasoning_trace=reasoning_trace,
        )
