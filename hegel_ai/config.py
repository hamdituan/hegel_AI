"""Centralized configuration management with Pydantic validation."""

import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Environment(str, Enum):
    DEVELOPMENT = "dev"
    PRODUCTION = "prod"
    TEST = "test"


class AgentConfig(BaseModel):
    """Debate agent configuration."""
    name: str = Field(..., min_length=1, max_length=50)
    concepts: str = Field(..., min_length=10)
    example_phrases: str = Field(..., min_length=10)
    system: str = Field(..., min_length=50)
    temperature: float = Field(..., ge=0.0, le=1.0)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip().title()

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        return round(v, 2)


class Config(BaseModel):
    """Application configuration."""

    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug_mode: bool = Field(default=False)

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.resolve()
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "sources"
    )
    vector_db_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "vector_db"
    )
    passage_path: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "passage.txt"
    )
    output_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "output"
    )
    log_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "logs"
    )

    use_ollama: bool = Field(default=True)
    ollama_model: str = Field(default="gemma3:1b")
    ollama_base_url: str = Field(default="http://localhost:11434")
    max_context_tokens: int = Field(default=8192, ge=1024, le=32768)
    llm_timeout: int = Field(default=60, ge=10, le=300)
    llm_max_retries: int = Field(default=3, ge=0, le=5)

    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu")
    embedding_normalize: bool = Field(default=True)

    num_rounds: int = Field(default=3, ge=1, le=10)
    use_moderator: bool = Field(default=True)
    agents: List[AgentConfig] = Field(default_factory=list)

    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_diversity: bool = Field(default=True)
    min_relevance_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    filter_front_matter: bool = Field(default=True)

    chunk_batch_size: int = Field(default=32, ge=8, le=128)
    max_chunks_per_doc: int = Field(default=500, ge=50, le=2000)

    citation_required: bool = Field(default=True)
    citation_retry_count: int = Field(default=2, ge=0, le=3)

    use_tree_of_thought: bool = Field(default=True)
    tot_num_branches: int = Field(default=2, ge=1, le=3)
    use_self_refinement: bool = Field(default=True)
    refinement_num_passes: int = Field(default=1, ge=1, le=2)
    use_dialectical_structure: bool = Field(default=True)
    show_reasoning_trace: bool = Field(default=True)

    class Config:
        arbitrary_types_allowed = True
        extra = 'ignore'

    @field_validator('agents')
    @classmethod
    def validate_agents(cls, v: List[AgentConfig]) -> List[AgentConfig]:
        """Validate agent list."""
        if not v:
            raise ValueError("At least one agent must be configured")

        names = [agent.name for agent in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Agent names must be unique. Duplicates found: {duplicates}")

        return v

    @model_validator(mode='after')
    def validate_all_paths(self) -> 'Config':
        return self

    def validate_vector_db(self) -> bool:
        """Validate vector database."""
        if not self.vector_db_dir.exists():
            raise FileNotFoundError(f"Vector database directory not found: {self.vector_db_dir}")

        db_file = self.vector_db_dir / "chroma.sqlite3"
        if not db_file.exists():
            raise FileNotFoundError(f"Chroma database file not found: {db_file}")

        return True

    def get_agent_by_name(self, name: str) -> Optional[AgentConfig]:
        for agent in self.agents:
            if agent.name.lower() == name.lower():
                return agent
        return None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode='json')

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "HEGEL AI CONFIGURATION SUMMARY",
            "=" * 60,
            f"Environment: {self.environment.value}",
            f"Debug Mode: {self.debug_mode}",
            "",
            "LLM Configuration:",
            f"  Model: {self.ollama_model}",
            f"  Base URL: {self.ollama_base_url}",
            f"  Max Context: {self.max_context_tokens} tokens",
            f"  Timeout: {self.llm_timeout}s",
            f"  Max Retries: {self.llm_max_retries}",
            "",
            "Embedding Configuration:",
            f"  Model: {self.embedding_model}",
            f"  Device: {self.embedding_device}",
            "",
            "Debate Configuration:",
            f"  Rounds: {self.num_rounds}",
            f"  Moderator: {'Enabled' if self.use_moderator else 'Disabled'}",
            f"  Agents: {len(self.agents)} ({', '.join(a.name for a in self.agents)})",
            "",
            "Retrieval Configuration:",
            f"  Top-K: {self.retrieval_top_k}",
            f"  Diversity: {'Enabled' if self.retrieval_diversity else 'Disabled'}",
            f"  Min Relevance: {self.min_relevance_threshold}",
            "",
            "Paths:",
            f"  Project Root: {self.project_root}",
            f"  Data Dir: {self.data_dir}",
            f"  Vector DB: {self.vector_db_dir}",
            f"  Passage: {self.passage_path}",
            f"  Output: {self.output_dir}",
            "=" * 60,
        ]
        return "\n".join(lines)


_config: Optional[Config] = None


def get_default_agents() -> List[AgentConfig]:
    return [
        AgentConfig(
            name="Hegelian",
            concepts="dialectic, Geist, Being/Nothing/Becoming, master/slave, actuality, rational/actual",
            example_phrases="The passage exemplifies the movement from Being to Nothing…; Geist reveals itself through this negation…; This dialectical tension shows…",
            system="""You are a Hegelian scholar. Your core concepts: dialectic, Geist, Being/Nothing/Becoming, master/slave, actuality.

**CRITICAL**: You MUST begin your response by quoting a provided excerpt using the exact format: "As [source filename] states: '...'"

Then analyse the quote in relation to Hegel's dialectic, Geist, and the given passage.

**Uniqueness**: Do not repeat arguments already made by other agents in this round. If you agree, add a new angle or counterpoint.

Keep your response to 1–2 paragraphs (max 250 words).""",
            temperature=0.5
        ),
        AgentConfig(
            name="Utilitarian",
            concepts="greatest happiness, utility, consequences, pleasure/pain, aggregate welfare, cost-benefit",
            example_phrases="The action maximizes utility for the greatest number…; The consequences produce more pleasure than pain…; A cost-benefit analysis reveals…",
            system="""You are a Utilitarian philosopher in the tradition of Bentham and Mill.
Your core principles: the greatest happiness principle, utility calculus, hedonism, and maximising aggregate well‑being.

**CRITICAL**: You MUST begin your response by quoting a provided excerpt using the exact format: "As [source filename] states: '...'"

Then analyse the quote in utilitarian terms: how does it relate to happiness, pleasure, pain, consequences, or the greatest good?

Use your key concepts: greatest happiness principle, utility calculus, hedonism, consequences, aggregate well‑being.

**Uniqueness**: Do not repeat arguments already made by other agents in this round. If you agree, add a new angle or counterpoint.

Keep your response to 1–2 paragraphs (max 250 words).""",
            temperature=0.5
        ),
        AgentConfig(
            name="Deconstructionist",
            concepts="binary oppositions, différance, aporia, logocentrism, supplement, trace, undecidability",
            example_phrases="The binary opposition between X and Y collapses…; The text deconstructs its own claim…; This aporia reveals the instability of…",
            system="""You are a Deconstructionist (Derrida). Your core concepts: binary oppositions, aporia, différance, supplement, logocentrism.

**CRITICAL**: You MUST begin your response by quoting a provided excerpt using the exact format: "As [source filename] states: '...'"

Then examine the binary oppositions, contradictions, and how the text undermines its own assumptions.

**Uniqueness**: Do not repeat arguments already made by other agents in this round. If you agree, add a new angle or counterpoint. Your analysis should be distinct from Postcolonial critiques.

Keep your response to 1–2 paragraphs (max 250 words).""",
            temperature=0.6
        ),
        AgentConfig(
            name="Postcolonial",
            concepts="subaltern, Orientalism, mimicry, hybridity, colonial discourse, Eurocentrism, othering",
            example_phrases="The construction of Africa as 'non‑historical' echoes Orientalist stereotypes…; This silencing of the subaltern voice…; The colonial discourse positions the West as the universal subject…",
            system="""You are a Postcolonial critic (following Fanon, Said, Spivak, etc.).
Your core concepts: subaltern, Orientalism, mimicry, hybridity, colonial discourse.

**CRITICAL**: You MUST begin your response by quoting a provided excerpt using the exact format: "As [source filename] states: '...'"

Then analyse how the passage reflects colonial power dynamics, Eurocentrism, or the erasure of subaltern voices.

Use your key concepts: subaltern, Orientalism, mimicry, hybridity, colonial discourse.

**Uniqueness**: Do not repeat arguments already made by other agents in this round. If you agree, add a new angle or counterpoint.

Keep your response to 1–2 paragraphs (max 250 words).""",
            temperature=0.5
        ),
    ]


def load_config(config_path: Optional[Path] = None, use_default_agents: bool = True) -> Config:
    import yaml

    config_dict: Dict[str, Any] = {
        'environment': os.getenv('HEGEL_ENV', 'dev'),
        'debug_mode': os.getenv('DEBUG', 'false').lower() == 'true',
        'ollama_model': os.getenv('OLLAMA_MODEL', 'gemma3:1b'),
        'ollama_base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        'num_rounds': int(os.getenv('NUM_ROUNDS', '3')),
    }

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"

    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config_dict.update(yaml_config)

    if use_default_agents and 'agents' not in config_dict:
        config_dict['agents'] = [agent.model_dump() for agent in get_default_agents()]

    return Config(**config_dict)


def get_config(reload: bool = False) -> Config:
    global _config

    if _config is None or reload:
        _config = load_config()

    return _config


def set_config(config: Config) -> None:
    global _config
    _config = config


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data" / "sources"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
PASSAGE_PATH = PROJECT_ROOT / "data" / "passage.txt"
