"""Agents package - Philosophical debate agents."""

from .base import Agent, AgentRegistry
from .hegelian import HegelianAgent
from .utilitarian import UtilitarianAgent
from .deconstructionist import DeconstructionistAgent
from .postcolonial import PostcolonialAgent

__all__ = [
    "Agent",
    "AgentRegistry",
    "HegelianAgent",
    "UtilitarianAgent",
    "DeconstructionistAgent",
    "PostcolonialAgent",
]
