"""
StatAgent - An intelligent statistical analysis toolkit.

StatAgent provides a comprehensive suite of tools for probabilistic modeling,
hypothesis testing, Bayesian inference, and regression analysis.
"""

__version__ = "0.1.0"
__author__ = "Navid Broumandfar"

from statagent.distributions import NegativeBinomialAnalyzer, SurvivalMixtureModel
from statagent.estimation import MethodOfMoments, BayesianEstimator
from statagent.inference import ZTest
from statagent.regression import PolynomialRegression

__all__ = [
    "NegativeBinomialAnalyzer",
    "SurvivalMixtureModel",
    "MethodOfMoments",
    "BayesianEstimator",
    "ZTest",
    "PolynomialRegression",
]

