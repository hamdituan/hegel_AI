"""Debate models for structured output."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Agent response."""
    agent_name: str = Field(..., description="Agent name")
    round_number: int = Field(..., ge=1, description="Debate round")
    citation_source: str = Field(default="UNKNOWN", description="Cited source file")
    citation_quote: str = Field(default="", description="Direct quote")
    analysis: str = Field(..., description="Analysis text")
    word_count: int = Field(..., ge=0, description="Word count")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    raw_response: str = Field(..., description="Original LLM response")
    validation_passed: bool = Field(default=False, description="Validation status")
    validation_errors: List[str] = Field(default_factory=list)
    tot_branches: List[Dict[str, Any]] = Field(default_factory=list)
    tot_selected_branch: Optional[Dict[str, Any]] = None
    refinement_passes: int = Field(default=0)
    refinement_improvements: List[str] = Field(default_factory=list)
    reasoning_trace: str = Field(default="")
    dialectical_stage: str = Field(default="thesis")

    @classmethod
    def parse_from_raw(
        cls,
        raw: str,
        agent_name: str,
        round_num: int,
        validate_citation: bool = True,
        retrieved_sources: Optional[List[str]] = None,
    ) -> "AgentResponse":
        import re

        citation_patterns = [
            r'As\s+\[([^\]]+)\]\s+states:\s*[\'"]([^\'"]+)[\'"]',
            r'As\s+\[([^\]]+)\]\s+states:',
            r'\[Source: ([^\]]+)\]',
        ]

        citation_source = "UNKNOWN"
        citation_quote = ""

        for pattern in citation_patterns:
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                citation_source = match.group(1).strip()
                if len(match.groups()) > 1:
                    citation_quote = match.group(2).strip()
                break

        analysis = raw
        for pattern in citation_patterns:
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                analysis = raw[match.end():].strip()
                break

        word_count = len(analysis.split())
        validation_errors = []
        validation_passed = True

        if validate_citation:
            if citation_source == "UNKNOWN":
                validation_errors.append("No citation found")
                validation_passed = False
            elif retrieved_sources:
                source_match = False
                for src in retrieved_sources:
                    if citation_source.lower() in src.lower() or src.lower() in citation_source.lower():
                        source_match = True
                        break

                if not source_match:
                    validation_errors.append(f"Cited source not in retrieved documents: {citation_source}")
                    validation_passed = False

        return cls(
            agent_name=agent_name,
            round_number=round_num,
            citation_source=citation_source,
            citation_quote=citation_quote,
            analysis=analysis,
            word_count=word_count,
            raw_response=raw,
            validation_passed=validation_passed,
            validation_errors=validation_errors,
        )


class DebateTurn(BaseModel):
    """Debate turn."""
    agent: str = Field(..., description="Agent name")
    round: int = Field(..., ge=1, description="Round number")
    response: AgentResponse = Field(..., description="Agent response")
    retrieval_metrics: Optional[Dict[str, Any]] = Field(default=None)
    validation_passed: bool = Field(default=False)
    validation_errors: List[str] = Field(default_factory=list)


class DebateRecord(BaseModel):
    """Debate record."""
    passage: str = Field(..., description="Target passage")
    annotations: str = Field(..., description="Passage annotations")
    turns: List[DebateTurn] = Field(default_factory=list)
    total_rounds: int = Field(..., ge=1)
    agents: List[str] = Field(default_factory=list)
    start_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = Field(default=None)
    total_duration_seconds: float = Field(default=0.0)

    def add_turn(self, turn: DebateTurn) -> None:
        self.turns.append(turn)

    def finalize(self, end_time: Optional[str] = None) -> None:
        self.end_time = end_time or datetime.now().isoformat()
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)
        self.total_duration_seconds = (end - start).total_seconds()

    def get_statistics(self) -> Dict[str, Any]:
        if not self.turns:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time) if self.end_time else datetime.now()
            duration = (end - start).total_seconds()
            
            return {
                "total_turns": 0,
                "total_words": 0,
                "avg_words_per_turn": 0,
                "citation_rate": 0.0,
                "validation_pass_rate": 0.0,
                "unique_citation_sources": 0,
                "duration_seconds": round(duration, 1),
            }

        total_words = sum(turn.response.word_count for turn in self.turns)
        citation_count = sum(
            1 for turn in self.turns
            if turn.response.citation_source != "UNKNOWN"
        )
        validation_pass_count = sum(
            1 for turn in self.turns
            if turn.validation_passed
        )
        unique_sources = set(
            turn.response.citation_source
            for turn in self.turns
            if turn.response.citation_source != "UNKNOWN"
        )

        duration = self.total_duration_seconds
        if duration == 0.0:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time) if self.end_time else datetime.now()
            duration = (end - start).total_seconds()

        return {
            "total_turns": len(self.turns),
            "total_words": total_words,
            "avg_words_per_turn": round(total_words / len(self.turns), 1),
            "citation_rate": round(citation_count / len(self.turns), 2),
            "validation_pass_rate": round(validation_pass_count / len(self.turns), 2),
            "unique_citation_sources": len(unique_sources),
            "duration_seconds": round(duration, 1),
        }

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

        return path

    @classmethod
    def load_json(cls, path: Path) -> "DebateRecord":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_transcript(self) -> str:
        lines = [
            "=" * 80,
            "HEGEL AI PHILOSOPHICAL DEBATE",
            "=" * 80,
            "",
            "PASSAGE:",
            self.passage,
            "",
            "ANNOTATIONS:",
            self.annotations,
            "",
            "-" * 80,
            "DEBATE",
            "-" * 80,
            "",
        ]

        current_round = 0
        for turn in self.turns:
            if turn.round != current_round:
                current_round = turn.round
                lines.append(f"\n{'=' * 40}\nROUND {current_round}\n{'=' * 40}\n")

            lines.append(f"\n{turn.agent} (Round {turn.round}):")
            lines.append(f"  Citation: [{turn.response.citation_source}]")

            if turn.response.validation_passed:
                lines.append("  Status: Valid")
            else:
                lines.append(f"  Status: Invalid - {', '.join(turn.response.validation_errors)}")

            lines.append(f"  Words: {turn.response.word_count}")
            lines.append("")
            lines.append(turn.response.analysis)
            lines.append("")

        lines.append("\n" + "=" * 80)
        lines.append("END OF DEBATE")
        lines.append("=" * 80)

        stats = self.get_statistics()
        lines.append("")
        lines.append("STATISTICS:")
        lines.append(f"  Total Turns: {stats['total_turns']}")
        lines.append(f"  Total Words: {stats['total_words']}")
        lines.append(f"  Avg Words/Turn: {stats['avg_words_per_turn']}")
        lines.append(f"  Citation Rate: {stats['citation_rate'] * 100:.0f}%")
        lines.append(f"  Validation Pass Rate: {stats['validation_pass_rate'] * 100:.0f}%")
        lines.append(f"  Unique Sources Cited: {stats['unique_citation_sources']}")
        lines.append(f"  Duration: {stats['duration_seconds']:.1f}s")

        return "\n".join(lines)

    def save_transcript(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)

        transcript = self.to_transcript()

        with open(path, "w", encoding="utf-8") as f:
            f.write(transcript)

        return path
