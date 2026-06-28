"""
StatAgent experimental agent layer.

This module provides an intelligent agent layer on top of StatAgent's statistical
tools, enabling rule-based or LLM-assisted method selection and interpretation.
"""

from statagent.agent.statistical_agent import StatisticalAgent
from statagent.agent.data_examiner import DataExaminer
from statagent.agent.reasoning_engine import ReasoningEngine
from statagent.agent.orchestrator import Orchestrator
from statagent.agent.interpreter import Interpreter

__all__ = [
    "StatisticalAgent",
    "DataExaminer",
    "ReasoningEngine",
    "Orchestrator",
    "Interpreter",
]
