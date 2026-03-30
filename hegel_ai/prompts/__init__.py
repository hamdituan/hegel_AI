"""
Advanced prompt engineering for Hegel AI.

Provides sophisticated prompting strategies:
- Tree of Thought (ToT)
- Self-Refinement
- Chain of Thought (CoT)
- Dialectical prompting
"""

from .tree_of_thought import TreeOfThought, ThoughtBranch
from .refinement import SelfRefinement, RefinementResult
from .dialectical import DialecticalPrompt, DialecticalStage

__all__ = [
    "TreeOfThought",
    "ThoughtBranch",
    "SelfRefinement",
    "RefinementResult",
    "DialecticalPrompt",
    "DialecticalStage",
]
