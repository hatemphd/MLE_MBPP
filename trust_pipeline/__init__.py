"""Agentic trust pipeline for AI-generated code on MBPP.

Labels come from hidden-test execution and static analysis (never from AI judgment); a classifier
trained on decision-time features produces APPROVED / REJECTED / NEEDS HUMAN REVIEW.
"""

from .config import Config

__all__ = ["Config"]
