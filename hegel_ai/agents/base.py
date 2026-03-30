"""Base agent classes for Hegel AI debate system."""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from hegel_ai.config import AgentConfig, get_config
from hegel_ai.llm.ollama_client import get_llm_client
from hegel_ai.logging_config import get_logger
from hegel_ai.debate.models import AgentResponse

logger = get_logger("agents.base")


@dataclass
class ThoughtBranch:
    """Lightweight thought branch for ToT."""
    id: int
    perspective: str
    argument: str = ""
    citation_source: str = ""
    citation_quote: str = ""
    score: float = 0.0


class TreeOfThought:
    """Lightweight Tree of Thought focused on grounding."""

    def __init__(self, num_branches: int = 2):
        self.num_branches = num_branches
        self._client = get_llm_client()

    def generate_branches(
        self,
        passage: str,
        agent_concepts: str,
        retrieved_excerpts: str,
        system_prompt: str,
    ) -> List[ThoughtBranch]:
        branches = []
        perspectives = self._get_perspectives(agent_concepts)

        for i, perspective in enumerate(perspectives[:self.num_branches]):
            branch = self._generate_branch(
                branch_id=i + 1,
                perspective=perspective,
                passage=passage,
                agent_concepts=agent_concepts,
                retrieved_excerpts=retrieved_excerpts,
                system_prompt=system_prompt,
            )
            branches.append(branch)

        return branches

    def _get_perspectives(self, concepts: str) -> List[str]:
        concept_list = [c.strip() for c in concepts.split(",")]
        perspectives = []

        if len(concept_list) >= 2:
            perspectives.append(f"Focus on {concept_list[0]} and {concept_list[1]}")
            perspectives.append(f"Emphasize {concept_list[-1]} and its implications")
        else:
            perspectives.append("Direct conceptual analysis")
            perspectives.append("Comparative interpretation")

        return perspectives

    def _generate_branch(
        self,
        branch_id: int,
        perspective: str,
        passage: str,
        agent_concepts: str,
        retrieved_excerpts: str,
        system_prompt: str,
    ) -> ThoughtBranch:
        prompt = f"""{system_prompt}

PERSPECTIVE: {perspective}

PASSAGE: "{passage}"

RETRIEVED EXCERPTS:
{retrieved_excerpts}

Generate a concise argument (100-150 words) that:
1. Starts with: "As [filename.txt] states: '[quote]'"
2. Analyzes using your philosophical concepts
3. Stays focused on {perspective}

RESPONSE:"""

        try:
            response = self._client.generate_with_retry(
                prompt=prompt,
                temperature=0.5,
                max_tokens=250,
            )

            citation_source, citation_quote = self._extract_citation(response)
            score = self._score_branch(response, citation_source, citation_quote, passage)

            return ThoughtBranch(
                id=branch_id,
                perspective=perspective,
                argument=response,
                citation_source=citation_source,
                citation_quote=citation_quote,
                score=score,
            )

        except Exception as e:
            logger.warning(f"Branch {branch_id} generation failed: {e}")
            return ThoughtBranch(
                id=branch_id,
                perspective=perspective,
                argument="",
                citation_source="UNKNOWN",
                citation_quote="",
                score=0.0,
            )

    def _extract_citation(self, text: str) -> tuple:
        patterns = [
            r'As\s+\[([^\]]+)\]\s+states:\s*["\']([^"\']+)["\']',
            r'As\s+\[([^\]]+)\]\s+states:',
            r'\[Source:\s*([^\]]+)\]',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip() if len(match.groups()) > 1 else ""
        return "UNKNOWN", ""

    def _score_branch(self, argument: str, citation_source: str, citation_quote: str, passage: str) -> float:
        score = 0.0

        if citation_source != "UNKNOWN":
            score += 0.4

        if len(citation_quote) >= 10:
            score += 0.3

        if len(argument.split()) >= 80:
            score += 0.2

        if any(word in argument.lower() for word in ["dialectic", "utility", "binary", "colonial", "subaltern"]):
            score += 0.1

        return score

    def select_best(self, branches: List[ThoughtBranch]) -> ThoughtBranch:
        valid = [b for b in branches if b.argument and b.citation_source != "UNKNOWN"]
        if not valid:
            valid = branches
        return max(valid, key=lambda b: b.score)

    def synthesize(
        self,
        branches: List[ThoughtBranch],
        selected: ThoughtBranch,
        agent_concepts: str,
        passage: str,
    ) -> str:
        if len(branches) < 2 or not selected.argument:
            return selected.argument

        other_insights = "\n".join([
            f"- {b.argument[:100]}..." for b in branches
            if b.id != selected.id and b.argument
        ])

        prompt = f"""Synthesize a coherent argument (150-200 words) by:
1. Using this primary argument:
{selected.argument}

2. Incorporating insights where they strengthen your case:
{other_insights}

3. Starting with citation: "As [{selected.citation_source}] states: '{selected.citation_quote}'"

4. Using these concepts: {agent_concepts}

5. Analyzing this passage: "{passage}"

SYNTHESIS:"""

        try:
            return self._client.generate_with_retry(
                prompt=prompt,
                temperature=0.4,
                max_tokens=350,
            )
        except Exception:
            return selected.argument


class SelfRefinement:
    """Lightweight self-refinement focused on citation compliance."""

    def __init__(self, num_passes: int = 1):
        self.num_passes = num_passes
        self._client = get_llm_client()

    def refine(
        self,
        argument: str,
        passage: str,
        agent_concepts: str,
        system_prompt: str,
        retrieved_excerpts: str,
    ) -> str:
        citation_source, citation_quote = self._extract_citation(argument)

        has_valid_citation = citation_source != "UNKNOWN" and len(citation_quote) >= 10
        word_count = len(argument.split())

        # Skip refinement if already compliant as per report recommendation 3.5
        if has_valid_citation and word_count >= 150:
            logger.debug("Skipping refinement: initial response is already compliant")
            return argument

        critique = self._critique(
            argument=argument,
            passage=passage,
            agent_concepts=agent_concepts,
            citation_source=citation_source,
            citation_quote=citation_quote,
        )

        return self._revise(
            original=argument,
            critique=critique,
            passage=passage,
            agent_concepts=agent_concepts,
            system_prompt=system_prompt,
            retrieved_excerpts=retrieved_excerpts,
        )

    def _extract_citation(self, text: str) -> tuple:
        patterns = [
            r'As\s+\[([^\]]+)\]\s+states:\s*["\']([^"\']+)["\']',
            r'\[Source:\s*([^\]]+)\]',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip() if len(match.groups()) > 1 else ""
        return "UNKNOWN", ""

    def _critique(
        self,
        argument: str,
        passage: str,
        agent_concepts: str,
        citation_source: str,
        citation_quote: str,
    ) -> Dict[str, Any]:
        issues = []

        if citation_source == "UNKNOWN":
            issues.append("Missing citation - must start with 'As [filename.txt] states: ...'")

        if len(citation_quote) < 10:
            issues.append(f"Citation quote too short ({len(citation_quote)} chars) - need 10+ words")

        word_count = len(argument.split())
        if word_count < 120:
            issues.append(f"Too short ({word_count} words) - expand to 150-200 words")

        if not any(word in argument.lower() for word in agent_concepts.lower().split(",")):
            issues.append("Not using philosophical concepts deeply enough")

        return {"issues": issues, "count": len(issues)}

    def _revise(
        self,
        original: str,
        critique: Dict[str, Any],
        passage: str,
        agent_concepts: str,
        system_prompt: str,
        retrieved_excerpts: str,
    ) -> str:
        if not critique["issues"]:
            return original

        prompt = f"""{system_prompt}

REVISE your argument to fix these issues:
{chr(10).join('- ' + issue for issue in critique['issues'])}

ORIGINAL ARGUMENT:
{original}

RETRIEVED EXCERPTS (cite one at the start):
{retrieved_excerpts}

PASSAGE: "{passage}"

CONCEPTS: {agent_concepts}

REVISED ARGUMENT (150-200 words, MUST start with proper citation):"""

        try:
            return self._client.generate_with_retry(
                prompt=prompt,
                temperature=0.5,
                max_tokens=400,
            )
        except Exception:
            return original


class DialecticalTracker:
    """Tracks dialectical progression across debate rounds."""

    def __init__(self):
        self.stages = ["thesis", "antithesis", "synthesis"]

    def get_stage(self, round_num: int, agent_position: int) -> str:
        total_positions = 4
        turn_index = (round_num - 1) * total_positions + agent_position
        return self.stages[turn_index % 3]

    def get_stage_instruction(self, stage: str, agent_name: str) -> str:
        instructions = {
            "thesis": f"Present your initial {agent_name} position. Establish your interpretive framework.",
            "antithesis": f"Challenge or complicate the thesis. Introduce tension or counter-argument.",
            "synthesis": f"Resolve tensions. Show how contradictions lead to higher understanding.",
        }
        return instructions.get(stage, "")


class Agent(ABC):
    """Base class for debate agents."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._client = get_llm_client()
        self._app_config = get_config()

        # Reduce ToT branches for weak agents as per report recommendations
        tot_branches = self._app_config.tot_num_branches
        if self.config.name.lower() in ["utilitarian", "postcolonial"]:
            tot_branches = 2
            logger.info(f"Using reduced ToT branches (2) for {self.config.name}")

        self._tot = TreeOfThought(
            num_branches=min(tot_branches, 3)
        ) if self._app_config.use_tree_of_thought else None

        self._refinement = SelfRefinement(
            num_passes=min(1, self._app_config.refinement_num_passes)
        ) if self._app_config.use_self_refinement else None

        self._dialectical = DialecticalTracker() if self._app_config.use_dialectical_structure else None

    @abstractmethod
    def build_prompt(
        self,
        passage: str,
        retrieved: List[Document],
        debate_history: str,
        round_num: int,
    ) -> str:
        pass

    def generate_response(
        self,
        passage: str,
        retrieved: List[Document],
        debate_history: str,
        round_num: int,
        temperature: Optional[float] = None,
    ) -> AgentResponse:
        retrieved_excerpts = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content[:500]}"
            for doc in retrieved
        ]) if retrieved else "[No excerpts retrieved]"

        retrieved_sources = [doc.metadata.get("source", "") for doc in retrieved]

        if temperature is None:
            temperature = self.config.temperature
            if round_num >= 2:
                temperature = min(1.0, temperature + 0.1)

        dialectical_stage = ""
        tot_branches = []
        tot_selected = None
        reasoning_trace = ""
        refinement_passes = 0
        refinement_improvements = []

        if self._dialectical:
            agent_idx = 0
            for i, turn_line in enumerate(debate_history.split("\n")):
                if any(agent.name.lower() in turn_line.lower() for agent in self._app_config.agents):
                    agent_idx = (agent_idx + 1) % 4
            dialectical_stage = self._dialectical.get_stage(round_num, agent_idx)

        # Extract previous responses this round for uniqueness reminder
        round_marker = f"ROUND {round_num}"
        current_round_history = ""
        if round_marker in debate_history:
            current_round_history = debate_history.split(round_marker)[-1].strip()
        
        uniqueness_reminder = ""
        if current_round_history:
            # Simple summaries of previous responses
            previous_agents = []
            for line in current_round_history.split("\n"):
                if ": " in line and line.split(":")[0].strip() in ["Hegelian", "Utilitarian", "Deconstructionist", "Postcolonial", "Moderator"]:
                    previous_agents.append(line.split(":")[0].strip())
            
            if previous_agents:
                uniqueness_reminder = f"\n\n**UNIQUENESS REMINDER**: Previous responses this round from: {', '.join(previous_agents)}. Ensure your contribution adds a new perspective and does not repeat their points."

        if self._tot:
            logger.info(f"Running lightweight ToT for {self.config.name}")
            branches = self._tot.generate_branches(
                passage=passage,
                agent_concepts=self.config.concepts,
                retrieved_excerpts=retrieved_excerpts,
                system_prompt=self.config.system.format(concepts=self.config.concepts) + uniqueness_reminder,
            )

            tot_branches = [
                {"id": b.id, "perspective": b.perspective, "score": b.score, "has_citation": b.citation_source != "UNKNOWN"}
                for b in branches
            ]

            selected = self._tot.select_best(branches)
            tot_selected = {
                "id": selected.id,
                "perspective": selected.perspective,
                "score": selected.score,
            }

            final_argument = self._tot.synthesize(
                branches=branches,
                selected=selected,
                agent_concepts=self.config.concepts,
                passage=passage,
            )

            reasoning_trace = f"ToT: {len(branches)} branches, selected #{selected.id} (score: {selected.score:.2f})"
            logger.info(f"ToT selected branch {selected.id} (score: {selected.score:.2f})")

        else:
            prompt = self.build_prompt(passage, retrieved, debate_history, round_num)
            try:
                final_argument = self._client.generate_with_retry(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=400,
                )
            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                final_argument = f"[Error: {e}]"

        if self._refinement and final_argument:
            logger.info(f"Running self-refinement for {self.config.name}")
            refined = self._refinement.refine(
                argument=final_argument,
                passage=passage,
                agent_concepts=self.config.concepts,
                system_prompt=self.config.system.format(concepts=self.config.concepts),
                retrieved_excerpts=retrieved_excerpts,
            )

            if refined and len(refined) > len(final_argument):
                refinement_passes = 1
                refinement_improvements = ["Improved citation or expanded argument"]
                final_argument = refined

        citation_source, citation_quote = self._extract_citation(final_argument)

        response = AgentResponse(
            agent_name=self.config.name,
            round_number=round_num,
            citation_source=citation_source,
            citation_quote=citation_quote,
            analysis=final_argument,
            word_count=len(final_argument.split()),
            raw_response=final_argument,
            validation_passed=False,
            validation_errors=[],
            tot_branches=tot_branches,
            tot_selected_branch=tot_selected,
            refinement_passes=refinement_passes,
            refinement_improvements=refinement_improvements,
            reasoning_trace=reasoning_trace,
            dialectical_stage=dialectical_stage,
        )

        if citation_source == "UNKNOWN":
            response.validation_errors.append("No citation found")
        elif retrieved_sources and not any(
            citation_source.lower() in src.lower() or src.lower() in citation_source.lower()
            for src in retrieved_sources
        ):
            response.validation_errors.append("Cited source not in retrieved documents")

        response.validation_passed = len(response.validation_errors) == 0

        return response

    def _extract_citation(self, text: str) -> tuple:
        patterns = [
            r'As\s+\[([^\]]+)\]\s+states:\s*["\']([^"\']+)["\']',
            r'As\s+"([^"]+)"\s+states:\s*["\']([^"\']+)["\']',
            r'\[Source:\s*([^\]]+)\]',
            r'According\s+to\s+\[([^\]]+)\]',
            r'"([^"]+\.txt)"\s*[:\s]+\s*["\']([^"\']+)["\']',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                source = match.group(1).strip()
                quote = match.group(2).strip() if len(match.groups()) > 1 else ""
                return source, quote

        lines = text.split("\n")
        for line in lines[:3]:
            line = line.strip()
            if line and not line.startswith(("**", "##", "###", "1.", "2.", "3.")):
                if ".txt" in line or "Source" in line or "As" in line:
                    match = re.search(r"([A-Za-z][\w\s\.]*\.txt)", line)
                    if match:
                        source = match.group(1).strip()
                        quote_match = re.search(r'["\']([^"\']{10,})["\']', line)
                        quote = quote_match.group(1).strip() if quote_match else ""
                        return source, quote

        return "UNKNOWN", ""

    def build_retrieval_query(
        self,
        passage: str,
        round_num: int,
    ) -> str:
        dialectical_instruction = ""
        if self._dialectical:
            stage = self._dialectical.get_stage(round_num, 0)
            if stage == "antithesis":
                dialectical_instruction = "Find passages that challenge or complicate the dominant interpretation."
            elif stage == "synthesis":
                dialectical_instruction = "Find passages that could help resolve tensions or show higher unity."

        return f"""The passage is: "{passage}"

Agent perspective: {self.config.name}
Key concepts: {self.config.concepts}

{dialectical_instruction}

Find excerpts that could support a {self.config.name} critique of this passage.
Focus on passages related to: {self.config.concepts}
"""


class AgentRegistry:
    """Registry for agent types."""

    _agents: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        def wrapper(agent_class: type) -> type:
            cls._agents[name.lower()] = agent_class
            logger.debug(f"Registered agent: {name}")
            return agent_class
        return wrapper

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        return cls._agents.get(name.lower())

    @classmethod
    def list_agents(cls) -> List[str]:
        return list(cls._agents.keys())

    @classmethod
    def create_agent(cls, name: str, config: AgentConfig) -> Optional[Agent]:
        agent_class = cls.get(name)
        if agent_class:
            return agent_class(config)
        logger.warning(f"Unknown agent type: {name}")
        return None
