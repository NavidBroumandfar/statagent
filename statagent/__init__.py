"""
StatAgent - An intelligent statistical analysis toolkit.

StatAgent provides a comprehensive suite of tools for probabilistic modeling,
hypothesis testing, Bayesian inference, and regression analysis.

New in Phase 2: Autonomous Agent layer for intelligent analysis!
"""

__version__ = "0.2.0"
__author__ = "Navid Broumandfar"

from statagent.distributions import NegativeBinomialAnalyzer, SurvivalMixtureModel
from statagent.estimation import MethodOfMoments, BayesianEstimator
from statagent.inference import ZTest
from statagent.regression import PolynomialRegression

# Agent layer (Phase 2)
try:
    from statagent.agent import StatisticalAgent
    _agent_available = True
except ImportError:
    _agent_available = False
    StatisticalAgent = None

__all__ = [
    "NegativeBinomialAnalyzer",
    "SurvivalMixtureModel",
    "MethodOfMoments",
    "BayesianEstimator",
    "ZTest",
    "PolynomialRegression",
    "StatisticalAgent",  # New autonomous agent
]

