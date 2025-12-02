"""
Bayesian parameter estimation.

This module implements Bayesian estimation with conjugate priors,
particularly for Gamma-Exponential models.
"""

import numpy as np
from typing import Dict, Optional


class BayesianEstimator:
    """
    Bayesian parameter estimator with Gamma prior.
    
    For exponential likelihood with Gamma prior:
    - Prior: theta ~ Gamma(alpha_0, beta_0)
    - Likelihood: X_i ~ Exp(theta)
    - Posterior: theta | data ~ Gamma(alpha_post, beta_post)
    
    Parameters
    ----------
    alpha_prior : float
        Prior shape parameter
    beta_prior : float
        Prior rate parameter
    
    Attributes
    ----------
    alpha_prior : float
        Prior shape parameter
    beta_prior : float
        Prior rate parameter
    
    Examples
    --------
    >>> estimator = BayesianEstimator(alpha_prior=48, beta_prior=92)
    >>> result = estimator.update_gamma_exponential(
    ...     sample_mean=74.77, n=10, k=3
    ... )
    >>> print(f"Posterior mean: {result['bayes_estimate']:.4f}")
    """
    
    def __init__(self, alpha_prior: float, beta_prior: float):
        """Initialize the Bayesian estimator."""
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("Prior parameters must be positive")
        
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
    
    def update_gamma_exponential(
        self,
        sample_mean: float,
        n: int,
        k: float = 1
    ) -> Dict[str, float]:
        """
        Update Gamma prior with exponential/Gamma likelihood.
        
        For k=1, this is standard exponential likelihood.
        For general k, this handles Gamma(k, theta) likelihood.
        
        Posterior parameters:
        - alpha_post = alpha_prior + k * n
        - beta_post = beta_prior + n * sample_mean
        
        Parameters
        ----------
        sample_mean : float
            Sample mean of observations
        n : int
            Sample size
        k : float, optional
            Shape parameter (default: 1 for exponential)
            
        Returns
        -------
        result : dict
            Dictionary containing:
            - alpha_post: Posterior shape parameter
            - beta_post: Posterior rate parameter
            - bayes_estimate: Posterior mean (Bayes estimator)
            - map_estimate: Posterior mode (MAP estimator)
            - mle_estimate: Maximum likelihood estimate
        """
        if n <= 0:
            raise ValueError("Sample size must be positive")
        if sample_mean <= 0:
            raise ValueError("Sample mean must be positive")
        if k <= 0:
            raise ValueError("Shape parameter k must be positive")
        
        # Posterior parameters
        alpha_post = self.alpha_prior + k * n
        beta_post = self.beta_prior + n * sample_mean
        
        # Point estimates
        bayes_estimate = alpha_post / beta_post  # Posterior mean
        map_estimate = (alpha_post - 1) / beta_post  # Posterior mode
        mle_estimate = k / sample_mean  # MLE from likelihood only
        
        return {
            "alpha_post": float(alpha_post),
            "beta_post": float(beta_post),
            "bayes_estimate": float(bayes_estimate),
            "map_estimate": float(map_estimate),
            "mle_estimate": float(mle_estimate)
        }
    
    def posterior_variance(self, alpha_post: float, beta_post: float) -> float:
        """
        Compute posterior variance.
        
        For Gamma(alpha, beta), Var(theta) = alpha / beta^2
        
        Parameters
        ----------
        alpha_post : float
            Posterior shape parameter
        beta_post : float
            Posterior rate parameter
            
        Returns
        -------
        variance : float
            Posterior variance
        """
        return alpha_post / (beta_post ** 2)
    
    def credible_interval(
        self,
        alpha_post: float,
        beta_post: float,
        confidence: float = 0.95
    ) -> tuple:
        """
        Compute Bayesian credible interval.
        
        Parameters
        ----------
        alpha_post : float
            Posterior shape parameter
        beta_post : float
            Posterior rate parameter
        confidence : float, optional
            Confidence level (default: 0.95)
            
        Returns
        -------
        interval : tuple
            (lower, upper) bounds of credible interval
        """
        from scipy.stats import gamma
        
        alpha_tail = (1 - confidence) / 2
        lower = gamma.ppf(alpha_tail, alpha_post, scale=1/beta_post)
        upper = gamma.ppf(1 - alpha_tail, alpha_post, scale=1/beta_post)
        
        return (lower, upper)
    
    def summary(
        self,
        sample_mean: float,
        n: int,
        k: float = 1,
        show_interval: bool = True
    ) -> str:
        """
        Generate a summary of Bayesian estimation.
        
        Parameters
        ----------
        sample_mean : float
            Sample mean
        n : int
            Sample size
        k : float, optional
            Shape parameter
        show_interval : bool, optional
            Whether to show credible interval
            
        Returns
        -------
        summary : str
            Formatted summary string
        """
        result = self.update_gamma_exponential(sample_mean, n, k)
        var = self.posterior_variance(result["alpha_post"], result["beta_post"])
        
        summary = f"""
Bayesian Estimation (Gamma Prior)
==================================
Prior:
  alpha_0 = {self.alpha_prior}
  beta_0 = {self.beta_prior}
  Prior mean = {self.alpha_prior/self.beta_prior:.4f}

Data:
  n = {n}
  Sample mean = {sample_mean:.4f}
  k = {k}

Posterior:
  alpha_post = {result['alpha_post']:.0f}
  beta_post = {result['beta_post']:.4f}
  Posterior mean = {result['bayes_estimate']:.6f}
  Posterior variance = {var:.8f}

Point Estimates:
  Bayes estimate (mean): {result['bayes_estimate']:.6f}
  MAP estimate (mode): {result['map_estimate']:.6f}
  MLE estimate: {result['mle_estimate']:.6f}
"""
        
        if show_interval:
            interval = self.credible_interval(
                result["alpha_post"],
                result["beta_post"]
            )
            summary += f"""
95% Credible Interval: [{interval[0]:.6f}, {interval[1]:.6f}]
"""
        
        return summary

