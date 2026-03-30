"""Base agent classes for Hegel AI debate system."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from hegel_ai.config import AgentConfig, get_config
from hegel_ai.llm.ollama_client import get_llm_client
from hegel_ai.logging_config import get_logger
from hegel_ai.debate.models import AgentResponse
from hegel_ai.prompts.tree_of_thought import TreeOfThought
from hegel_ai.prompts.refinement import SelfRefinement

logger = get_logger("agents.base")


class Agent(ABC):
    """Base class for debate agents."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._client = get_llm_client()
        self._app_config = get_config()

        self._tot = TreeOfThought(
            num_branches=self._app_config.tot_num_branches,
            temperature_base=self.config.temperature,
        ) if self._app_config.use_tree_of_thought else None

        self._refinement = SelfRefinement(
            num_passes=self._app_config.refinement_num_passes,
        ) if self._app_config.use_self_refinement else None

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
        import re

        retrieved_excerpts = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content[:400]}"
            for doc in retrieved
        ]) if retrieved else "[No excerpts retrieved]"

        retrieved_sources = [doc.metadata.get("source", "") for doc in retrieved]

        if temperature is None:
            temperature = self.config.temperature
            if round_num >= 2:
                temperature = min(1.0, temperature + 0.1)

        tot_branches = []
        tot_selected = None
        reasoning_trace = ""
        refinement_passes = 0
        refinement_improvements = []
        final_argument = ""

        if self._tot:
            logger.info(f"Running Tree of Thought for {self.config.name}")
            tot_result = self._tot.run(
                passage=passage,
                agent_concepts=self.config.concepts,
                retrieved_excerpts=retrieved_excerpts,
                debate_context=debate_history,
            )

            tot_branches = [
                {
                    "id": b.id,
                    "perspective": b.perspective,
                    "reasoning": b.reasoning,
                    "score": b.score,
                    "evaluation": b.evaluation.value,
                }
                for b in tot_result.branches
            ]
            tot_selected = {
                "id": tot_result.selected_branch.id,
                "perspective": tot_result.selected_branch.perspective,
                "conclusion": tot_result.selected_branch.conclusion,
                "score": tot_result.selected_branch.score,
            }
            reasoning_trace = tot_result.reasoning_trace
            final_argument = tot_result.synthesis
            logger.info(f"ToT selected branch {tot_selected['id']} (score: {tot_selected['score']:.2f})")

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
            citation_source, citation_quote = self._extract_citation(final_argument)

            logger.info(f"Running self-refinement for {self.config.name}")
            refinement_result = self._refinement.refine(
                initial_argument=final_argument,
                passage=passage,
                agent_concepts=self.config.concepts,
                citation_source=citation_source,
                citation_quote=citation_quote,
            )

            final_argument = refinement_result.refined
            refinement_passes = refinement_result.num_passes
            refinement_improvements = refinement_result.improvements_made
            logger.info(f"Refinement complete: {len(refinement_improvements)} improvements")

        response = AgentResponse(
            agent_name=self.config.name,
            round_number=round_num,
            citation_source="UNKNOWN",
            citation_quote="",
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
        )

        citation_patterns = [
            r'As \[([^\]]+)\] states: [\'"]([^\'"]+)[\'"]',
            r'\[Source: ([^\]]+)\]',
        ]
        for pattern in citation_patterns:
            match = re.search(pattern, final_argument, re.IGNORECASE)
            if match:
                response.citation_source = match.group(1).strip()
                if len(match.groups()) > 1:
                    response.citation_quote = match.group(2).strip()
                break

        if response.citation_source == "UNKNOWN":
            response.validation_errors.append("No citation found")
        elif retrieved_sources and not any(
            response.citation_source.lower() in src.lower() or src.lower() in response.citation_source.lower()
            for src in retrieved_sources
        ):
            response.validation_errors.append("Cited source not in retrieved documents")

        response.validation_passed = len(response.validation_errors) == 0
        response.raw_response = final_argument

        return response

    def _extract_citation(self, text: str) -> tuple:
        import re
        patterns = [
            r'As \[([^\]]+)\] states: [\'"]([^\'"]+)[\'"]',
            r'\[Source: ([^\]]+)\]',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                source = match.group(1).strip()
                quote = match.group(2).strip() if len(match.groups()) > 1 else ""
                return source, quote
        return "UNKNOWN", ""

    def build_retrieval_query(
        self,
        passage: str,
        round_num: int,
    ) -> str:
        return f"""The passage is: "{passage}"

Agent perspective: {self.config.name}
Key concepts: {self.config.concepts}

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
