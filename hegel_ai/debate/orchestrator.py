"""Debate orchestrator for Hegel AI - multi-agent philosophical debates."""

import logging
import time
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from hegel_ai.agents.base import Agent, AgentRegistry
from hegel_ai.agents.hegelian import HegelianAgent
from hegel_ai.agents.utilitarian import UtilitarianAgent
from hegel_ai.agents.deconstructionist import DeconstructionistAgent
from hegel_ai.agents.postcolonial import PostcolonialAgent
from hegel_ai.config import Config, get_config, get_default_agents
from hegel_ai.debate.models import DebateRecord, DebateTurn, AgentResponse
from hegel_ai.debate.moderator import Moderator
from hegel_ai.llm.ollama_client import get_llm_client
from hegel_ai.logging_config import setup_logging, get_logger
from hegel_ai.output.manager import OutputManager
from hegel_ai.retrieval.vector_store import load_vector_store, retrieve_with_metrics

logger = get_logger("debate.orchestrator")


class DebateOrchestrator:
    """Debate orchestrator."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self._agents: List[Agent] = []
        self._moderator: Optional[Moderator] = None
        self._output_manager: Optional[OutputManager] = None

        self._initialize_agents()
        self._initialize_moderator()
        self._initialize_output_manager()

    def _initialize_agents(self) -> None:
        self._agents = []

        for agent_config in self.config.agents:
            agent = AgentRegistry.create_agent(agent_config.name, agent_config)
            if agent:
                self._agents.append(agent)
                logger.info(f"Initialized agent: {agent_config.name}")
            else:
                logger.warning(f"Failed to initialize agent: {agent_config.name}")

        if not self._agents:
            raise ValueError("No agents could be initialized")

        logger.info(f"Initialized {len(self._agents)} agents")

    def _initialize_moderator(self) -> None:
        if self.config.use_moderator:
            self._moderator = Moderator(temperature=0.4)
            logger.info("Moderator initialized")
        else:
            logger.info("Moderator disabled")

    def _initialize_output_manager(self) -> None:
        self._output_manager = OutputManager(self.config.output_dir)

    def run(
        self,
        passage: Optional[str] = None,
        passage_path: Optional[Path] = None,
        save_output: bool = True,
    ) -> DebateRecord:
        start_time = time.time()

        if passage is None:
            path = passage_path or self.config.passage_path
            if not path.exists():
                raise FileNotFoundError(f"Passage file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                passage = f.read().strip()

        if not passage:
            raise ValueError("Passage is empty")

        logger.info(f"Passage: {passage[:100]}...")

        vectorstore = load_vector_store(self.config.vector_db_dir)
        if vectorstore is None:
            raise RuntimeError(
                "Failed to load vector store. Run create_vector_db.py first."
            )

        annotations = self._annotate_passage(passage)
        logger.info(f"Annotations:\n{annotations[:300]}...")

        debate_record = DebateRecord(
            passage=passage,
            annotations=annotations,
            total_rounds=self.config.num_rounds,
            agents=[agent.config.name for agent in self._agents],
        )

        debate_history = f"Passage:\n{passage}\n\nAnnotations:\n{annotations}\n\n"

        total_turns = self.config.num_rounds * len(self._agents)
        if self.config.use_moderator:
            total_turns += self.config.num_rounds - 1

        with tqdm(total=total_turns, desc="Debate Progress", unit="turn") as pbar:
            for round_num in range(1, self.config.num_rounds + 1):
                logger.info(f"\n{'='*40}\nROUND {round_num}\n{'='*40}")

                for agent in self._agents:
                    turn = self._run_agent_turn(
                        agent=agent,
                        round_num=round_num,
                        passage=passage,
                        vectorstore=vectorstore,
                        debate_history=debate_history,
                        pbar=pbar,
                    )

                    if turn:
                        debate_record.add_turn(turn)
                        debate_history += f"{turn.agent}: {turn.response.raw_response}\n\n"

                    pbar.update(1)

                if (
                    self.config.use_moderator
                    and self._moderator
                    and round_num < self.config.num_rounds
                ):
                    turn = self._run_moderator_turn(
                        round_num=round_num,
                        debate_history=debate_history,
                        pbar=pbar,
                    )

                    if turn:
                        debate_history += f"Moderator: {turn.response.raw_response}\n\n"
                        pbar.update(1)

        debate_record.finalize()

        if save_output and self._output_manager:
            paths = self._output_manager.save_debate(debate_record)
            logger.info(f"\nOutput files saved:")
            for format_name, path in paths.items():
                logger.info(f"  {format_name}: {path}")

        stats = debate_record.get_statistics()
        logger.info("\n" + "=" * 60)
        logger.info("DEBATE COMPLETE")
        logger.info(f"  Total Turns: {stats['total_turns']}")
        logger.info(f"  Citation Rate: {stats['citation_rate'] * 100:.0f}%")
        logger.info(f"  Validation Pass Rate: {stats['validation_pass_rate'] * 100:.0f}%")
        logger.info(f"  Duration: {stats['duration_seconds']:.1f}s")
        logger.info("=" * 60)

        return debate_record

    def _run_agent_turn(
        self,
        agent: Agent,
        round_num: int,
        passage: str,
        vectorstore,
        debate_history: str,
        pbar,
    ) -> Optional[DebateTurn]:
        pbar.set_description(f"Round {round_num} - {agent.config.name}")

        try:
            query = agent.build_retrieval_query(passage, round_num)

            retrieved, metrics = retrieve_with_metrics(
                vectorstore=vectorstore,
                query=query,
                top_k=self.config.retrieval_top_k,
                min_relevance_threshold=self.config.min_relevance_threshold,
                enforce_diversity=self.config.retrieval_diversity,
                filter_front_matter=self.config.filter_front_matter,
            )

            pbar.write(f"\n[DEBUG] Retrieved for {agent.config.name}:")
            for i, doc in enumerate(retrieved):
                source = doc.metadata.get("source", "unknown")
                preview = doc.page_content[:150].replace("\n", " ")
                pbar.write(f"  {i+1}. {source} - {preview}...")

            response = self._generate_with_citation_retry(
                agent=agent,
                passage=passage,
                retrieved=retrieved,
                debate_history=debate_history,
                round_num=round_num,
                pbar=pbar,
            )

            # Check for repetition
            similarity = self._check_repetition(response.raw_response, debate_history, round_num)
            if similarity > 0.7:
                logger.warning(f"High repetition detected for {agent.config.name} (sim: {similarity:.2f}). Retrying.")
                response = self._generate_with_citation_retry(
                    agent=agent,
                    passage=passage,
                    retrieved=retrieved,
                    debate_history=debate_history + f"\n\n**WARNING**: Your previous attempt was too similar to other responses. Please provide a fresh perspective.",
                    round_num=round_num,
                    pbar=pbar,
                )

            turn = DebateTurn(
                agent=agent.config.name,
                round=round_num,
                response=response,
                retrieval_metrics=metrics.to_dict() if metrics else None,
                validation_passed=response.validation_passed,
                validation_errors=response.validation_errors,
            )

            status = "Valid" if response.validation_passed else "Invalid"
            pbar.write(f"\n{agent.config.name} (Round {round_num}):")
            pbar.write(f"   Citation: [{response.citation_source}] [{status}]")
            pbar.write(f"   Words: {response.word_count}")
            pbar.write(f"\n{response.analysis}\n")

            return turn

        except Exception as e:
            logger.error(f"Agent turn failed for {agent.config.name}: {e}")
            pbar.write(f"Error for {agent.config.name}: {e}")
            return None

    def _generate_with_citation_retry(
        self,
        agent: Agent,
        passage: str,
        retrieved: list,
        debate_history: str,
        round_num: int,
        pbar,
    ) -> AgentResponse:
        response = agent.generate_response(
            passage=passage,
            retrieved=retrieved,
            debate_history=debate_history,
            round_num=round_num,
        )

        retry_count = 0
        while (
            not response.validation_passed
            and retry_count < self.config.citation_retry_count
        ):
            pbar.write(
                f"{agent.config.name} did not cite properly. "
                f"Retrying ({retry_count + 1}/{self.config.citation_retry_count})..."
            )

            retry_prompt = self._build_retry_prompt(
                agent=agent,
                passage=passage,
                retrieved=retrieved,
                debate_history=debate_history,
                round_num=round_num,
                previous_response=response.raw_response,
            )

            try:
                from hegel_ai.llm.ollama_client import get_llm_client
                client = get_llm_client()
                raw_response = client.generate_with_retry(
                    prompt=retry_prompt,
                    temperature=min(1.0, agent.config.temperature + 0.1),
                    max_tokens=400,
                )

                retrieved_sources = [
                    doc.metadata.get("source", "") for doc in retrieved
                ]
                response = AgentResponse.parse_from_raw(
                    raw=raw_response,
                    agent_name=agent.config.name,
                    round_num=round_num,
                    validate_citation=True,
                    retrieved_sources=retrieved_sources,
                )

            except Exception as e:
                logger.error(f"Retry failed for {agent.config.name}: {e}")
                break

            retry_count += 1

        return response

    def _build_retry_prompt(
        self,
        agent: Agent,
        passage: str,
        retrieved: list,
        debate_history: str,
        round_num: int,
        previous_response: str,
    ) -> str:
        grounding_text = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content[:500]}..."
            for doc in retrieved
        ]) if retrieved else "[ERROR: No excerpts retrieved]"

        return f"""{agent.config.system.format(concepts=agent.config.concepts)}

**CRITICAL CITATION REQUIREMENT:**

Your previous response did not properly cite a source or was too repetitive. You MUST start with:

"As [source filename] states: '...'"

Example: "As [Hegel_Philosophy_of_History.txt] states: 'Spirit is self-contained existence.'"

RETRIEVED EXCERPTS (YOU MUST CITE ONE):
{grounding_text}

DEBATE HISTORY:
{debate_history[-2000:]}

YOUR PREVIOUS RESPONSE:
{previous_response[:500]}...

YOUR NEW RESPONSE (MUST BE UNIQUE AND START WITH PROPER CITATION):"""

    def _check_repetition(self, response: str, history: str, round_num: int) -> float:
        """Check for repetition against previous responses in the current round."""
        import re
        
        # Extract current round context
        round_marker = f"ROUND {round_num}"
        if round_marker not in history:
            return 0.0
            
        current_round_text = history.split(round_marker)[-1].strip()
        if not current_round_text:
            return 0.0
            
        # Helper to get sentences
        def get_sentences(text: str) -> set:
            # Simple sentence tokenizer
            sents = re.split(r'[.!?]\s+', text.lower())
            return {s.strip() for s in sents if len(s.strip()) > 20}
            
        res_sents = get_sentences(response)
        hist_sents = get_sentences(current_round_text)
        
        if not res_sents or not hist_sents:
            return 0.0
            
        intersection = res_sents.intersection(hist_sents)
        jaccard = len(intersection) / len(res_sents)
        
        # Also check for internal repetition
        words = response.lower().split()
        if len(words) > 50:
            for i in range(len(words) - 10):
                phrase = " ".join(words[i:i+10])
                if response.lower().count(phrase) > 1:
                    logger.warning(f"Internal repetition detected: '{phrase}'")
                    return 0.8  # Force retry
                    
        return jaccard

    def _run_moderator_turn(
        self,
        round_num: int,
        debate_history: str,
        pbar,
    ) -> Optional[DebateTurn]:
        pbar.set_description(f"Round {round_num} - Moderator")
        pbar.write("\nMODERATOR - Summarizing...")

        try:
            if not self._moderator:
                return None

            summary = self._moderator.summarize(
                debate_history=debate_history,
                agents=self.config.agents,
            )

            response = AgentResponse(
                agent_name="Moderator",
                round_number=round_num,
                citation_source="N/A",
                citation_quote="",
                analysis=summary,
                word_count=len(summary.split()),
                raw_response=summary,
                validation_passed=True,
            )

            turn = DebateTurn(
                agent="Moderator",
                round=round_num,
                response=response,
                validation_passed=True,
            )

            pbar.write(f"\nModerator:\n{summary}\n")
            return turn

        except Exception as e:
            logger.error(f"Moderator turn failed: {e}")
            pbar.write(f"Moderator error: {e}")
            return None

    def _annotate_passage(self, passage: str) -> str:
        from hegel_ai.llm.ollama_client import get_llm_client

        system = """You are a neutral literary annotation agent. Analyze the passage and output:
- **Key phrases**: list of 3-5 important phrases
- **Binary oppositions**: any implicit contrasts
- **Recurring motifs**: themes or patterns
- **Structural elements**: narrative voice, syntax, rhetorical devices
Be concise and objective."""

        prompt = f"{system}\n\nPassage:\n{passage}"

        try:
            client = get_llm_client()
            return client.generate_with_retry(
                prompt=prompt,
                temperature=0.2,
                max_tokens=300,
            )
        except Exception as e:
            logger.warning(f"Annotation failed: {e}")
            return "[Annotations unavailable]"


def run_debate(
    config: Optional[Config] = None,
    passage: Optional[str] = None,
    passage_path: Optional[Path] = None,
    save_output: bool = True,
) -> DebateRecord:
    setup_logging()

    orchestrator = DebateOrchestrator(config)
    return orchestrator.run(
        passage=passage,
        passage_path=passage_path,
        save_output=save_output,
    )


def main():
    setup_logging()

    try:
        run_debate()
    except Exception as e:
        logger.exception(f"Debate failed: {e}")
        raise


if __name__ == "__main__":
    main()
