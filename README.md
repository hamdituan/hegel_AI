# Hegel AI

A multi-agent philosophical debate system powered by RAG (Retrieval-Augmented Generation) and advanced prompting techniques.

## Overview

Hegel AI orchestrates structured debates between AI agents, each representing a distinct philosophical perspective (Hegelian, Utilitarian, Deconstructionist, Postcolonial). The system analyzes philosophical passages by retrieving relevant excerpts from Hegel's corpus and generating cited, structured arguments through:

- **Tree of Thought prompting** – Each agent explores multiple reasoning paths before committing to an argument
- **Self-Refinement** – Agents critique and revise their arguments through iterative passes
- **Dialectical structure** – Thesis-antithesis-synthesis debate format
- **Citation enforcement** – Mandatory source citation with validation and retry logic

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Debate** | 4 philosophical agents debate passages with distinct theoretical perspectives |
| **Tree of Thought** | 3 reasoning branches explored per agent, evaluated, and synthesized |
| **Self-Refinement** | 2 passes of critique and revision per argument |
| **RAG Pipeline** | Semantic chunking, embedding-based retrieval, MMR diversity |
| **Citation Validation** | Regex + source verification with automatic retry |
| **Structured Output** | JSON records, human-readable transcripts, statistics |
| **Moderator** | Summarizes disagreements and poses new questions between rounds |

## Quick Start

### Prerequisites

- Python 3.10+
- Ollama running locally with `gemma3:1b` model
- pip package manager

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Hegel_AI

# Install dependencies
pip install -r requirements.txt

# Download Ollama model
ollama pull gemma3:1b
```

### Usage

**1. Create the vector database** (one-time setup)

```bash
python run_vector_db.py
```

This processes 5 Hegel texts from `data/sources/hegel/` and creates a searchable vector store in `vector_db/`.

**2. Run a debate**

```bash
python run_debate.py
```

The system will:
1. Load the passage from `data/passage.txt`
2. Generate annotations
3. Run 3 rounds of debate between 4 agents
4. Save output to `output/` directory

### Example Output

Output files are saved to `output/`:
- `debate_YYYYMMDD_HHMMSS_<passage>.json` – Full structured record
- `debate_YYYYMMDD_HHMMSS_<passage>.txt` – Human-readable transcript
- `debate_YYYYMMDD_HHMMSS_<passage>_stats.txt` – Statistics

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Entry Points                              │
│              run_vector_db.py    run_debate.py                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      hegel_ai/ Package                           │
├─────────────────────────────────────────────────────────────────┤
│  config.py          │ Configuration management (Pydantic)       │
│  logging_config.py  │ Centralized logging                       │
├─────────────────────────────────────────────────────────────────┤
│  agents/            │ Philosophical agents                      │
│    ├── base.py      │ Base class with ToT + Refinement          │
│    ├── hegelian.py  │ Hegelian scholar                          │
│    ├── utilitarian.py│ Utilitarian philosopher                  │
│    ├── deconstructionist.py │ Deconstructionist critic          │
│    └── postcolonial.py    │ Postcolonial critic                 │
├─────────────────────────────────────────────────────────────────┤
│  prompts/           │ Advanced prompting strategies             │
│    ├── tree_of_thought.py │ Tree of Thought implementation     │
│    ├── refinement.py    │ Self-refinement loop                 │
│    └── dialectical.py   │ Thesis-antithesis-synthesis          │
├─────────────────────────────────────────────────────────────────┤
│  retrieval/         │ RAG pipeline                              │
│    ├── chunking.py  │ Semantic document chunking               │
│    ├── embeddings.py│ Embedding model management               │
│    ├── vector_store.py│ ChromaDB operations                    │
│    └── metrics.py   │ Retrieval quality metrics                │
├─────────────────────────────────────────────────────────────────┤
│  llm/               │ LLM client abstraction                    │
│    ├── client.py    │ Abstract base class                       │
│    └── ollama_client.py│ Ollama API integration                │
├─────────────────────────────────────────────────────────────────┤
│  debate/            │ Debate orchestration                      │
│    ├── orchestrator.py│ Main debate loop                       │
│    ├── moderator.py │ Moderator logic                          │
│    └── models.py    │ Pydantic models for structured output    │
├─────────────────────────────────────────────────────────────────┤
│  output/            │ Output management                         │
│    ├── manager.py   │ File generation                           │
│    └── analysis.py  │ Debate analysis                           │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

Edit `hegel_ai/config.py` or create `config.yaml` in project root:

```yaml
# Number of debate rounds
num_rounds: 3

# Enable/disable moderator
use_moderator: true

# Advanced prompting
use_tree_of_thought: true
tot_num_branches: 3
use_self_refinement: true
refinement_num_passes: 2
use_dialectical_structure: true

# LLM settings
ollama_model: "gemma3:1b"
ollama_base_url: "http://localhost:11434"
llm_timeout: 60
```

## Data Flow

```
1. USER INPUT (passage.txt)
         │
         ▼
2. ANNOTATION (LLM generates key phrases, motifs)
         │
         ▼
3. RETRIEVAL QUERY (agent builds query from concepts)
         │
         ▼
4. VECTOR SEARCH (ChromaDB similarity search with MMR)
         │
         ▼
5. TREE OF THOUGHT (3 branches generated + evaluated)
         │
         ▼
6. BRANCH SELECTION (best branch by score)
         │
         ▼
7. SELF-REFINEMENT (2 passes of critique + revision)
         │
         ▼
8. CITATION VALIDATION (regex + source verification)
         │
         ▼
9. DEBATE RECORD (Pydantic model into JSON + TXT output)
```

## Project Structure

```
Hegel_AI/
├── run_vector_db.py          # Create vector database
├── run_debate.py             # Run philosophical debate
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Package configuration
├── .gitignore               # Git ignore rules
├── data/
│   ├── passage.txt          # Target passage for analysis
│   └── sources/hegel/       # Hegel texts (5 files)
├── vector_db/               # Generated vector store [.gitignore]
├── output/                  # Debate transcripts
├── logs/                    # Log files [.gitignore]
└── hegel_ai/                # Main package
    ├── config.py
    ├── logging_config.py
    ├── agents/
    ├── prompts/
    ├── retrieval/
    ├── llm/
    ├── debate/
    └── output/
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| langchain | >=0.3.0 | RAG framework |
| chromadb | >=0.5.0 | Vector database |
| sentence-transformers | >=2.7.0 | Embeddings |
| ollama | >=0.3.0 | LLM inference |
| pydantic | >=2.0.0 | Configuration validation |
| nltk | >=3.9 | Text tokenization |
| tqdm | >=4.66.0 | Progress bars |

## Performance

| Metric | Value |
|--------|-------|
| Vector DB chunks | ~8,673 |
| Debate duration | ~3-5 minutes (3 rounds, 4 agents) |
| ToT branches per agent | 3 |
| Refinement passes | 2 |
| Citation rate | ~90%+ (with retry logic) |

## Advanced Usage

### Change the Passage

Edit `data/passage.txt` with any text you want analyzed:

```
"Your philosophical passage here..."
```

### Add a New Agent

1. Create `hegel_ai/agents/Confucian.py`:
```python
from hegel_ai.agents.base import Agent, AgentRegistry

@AgentRegistry.register("confucian")
class ConfucianAgent(Agent):
    def build_prompt(self, passage, retrieved, debate_history, round_num):
        # Implement prompt logic
        return prompt
```

2. Add to `hegel_ai/config.py` `get_default_agents()`:
```python
AgentConfig(
    name="Confucian",
    concepts="ren, yi, li, zhi, xin",
    example_phrases="The passage invokes order...",
    system="""You are a Confucian philosopher...""",
    temperature=0.5
)
```

### Disable Advanced Prompting (Faster)

```python
# In hegel_ai/config.py
use_tree_of_thought: bool = False
use_self_refinement: bool = False
```

Reduces debate time by ~3-5x with lower argument quality.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub.
