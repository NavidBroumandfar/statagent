"""
Method of Moments estimation.

This module implements the Method of Moments (MoM) estimator for various
probability distributions, particularly for Gamma distributions.
"""

import numpy as np
from typing import Dict, Optional


class MethodOfMoments:
    """
    Method of Moments estimator.
    
    The Method of Moments estimates distribution parameters by equating
    sample moments to theoretical moments and solving for the parameters.
    
    Parameters
    ----------
    data : array_like
        Observed sample data
    
    Attributes
    ----------
    data : np.ndarray
        Sample data
    n : int
        Sample size
    
    Examples
    --------
    >>> data = np.array([9, 18, 23, 41, 8])
    >>> mom = MethodOfMoments(data)
    >>> theta = mom.estimate_gamma_theta(k=2)
    >>> print(f"Estimated theta: {theta:.2f}")
    """
    
    def __init__(self, data: np.ndarray):
        """Initialize the Method of Moments estimator."""
        self.data = np.asarray(data, dtype=float)
        self.n = len(self.data)
        
        if self.n == 0:
            raise ValueError("Data array cannot be empty")
    
    def sample_mean(self) -> float:
        """
        Compute sample mean.
        
        Returns
        -------
        mean : float
            Sample mean
        """
        return float(np.mean(self.data))
    
    def sample_variance(self, ddof: int = 1) -> float:
        """
        Compute sample variance.
        
        Parameters
        ----------
        ddof : int, optional
            Degrees of freedom (default: 1 for unbiased estimate)
            
        Returns
        -------
        variance : float
            Sample variance
        """
        return float(np.var(self.data, ddof=ddof))
    
    def estimate_gamma_theta(self, k: float) -> float:
        """
        Estimate theta parameter for Gamma(k, theta) distribution.
        
        For a Gamma distribution with known shape parameter k and unknown
        scale parameter theta, the MoM estimator is:
        theta_hat = sample_mean / k
        
        Parameters
        ----------
        k : float
            Known shape parameter
            
        Returns
        -------
        theta_hat : float
            Estimated scale parameter
        """
        if k <= 0:
            raise ValueError("Shape parameter k must be positive")
        
        return self.sample_mean() / k
    
    def expected_value_gamma(self, k: float, theta: Optional[float] = None) -> float:
        """
        Compute expected value for Gamma distribution.
        
        E[X] = k * theta
        
        Parameters
        ----------
        k : float
            Shape parameter
        theta : float, optional
            Scale parameter (if None, uses estimated theta)
            
        Returns
        -------
        expected : float
            Expected value
        """
        if theta is None:
            theta = self.estimate_gamma_theta(k)
        
        return k * theta
    
    def estimate_gamma_both_params(self) -> Dict[str, float]:
        """
        Estimate both k and theta for Gamma distribution using MoM.
        
        For Gamma(k, theta):
        - E[X] = k * theta
        - Var(X) = k * theta^2
        
        Solving these equations:
        - k = E[X]^2 / Var(X)
        - theta = Var(X) / E[X]
        
        Returns
        -------
        params : dict
            Dictionary with keys 'k' and 'theta'
        """
        mean = self.sample_mean()
        var = self.sample_variance()
        
        if var <= 0:
            raise ValueError("Sample variance must be positive")
        
        k_hat = mean**2 / var
        theta_hat = var / mean
        
        return {"k": float(k_hat), "theta": float(theta_hat)}
    
    def summary(self, k: Optional[float] = None) -> str:
        """
        Generate a summary of the estimation.
        
        Parameters
        ----------
        k : float, optional
            If provided, estimates theta for given k
            
        Returns
        -------
        summary : str
            Formatted summary string
        """
        mean = self.sample_mean()
        var = self.sample_variance()
        
        summary = f"""
Method of Moments Estimation
=============================
Sample Statistics:
  n: {self.n}
  Sample mean: {mean:.4f}
  Sample variance: {var:.4f}
  Sample std: {np.sqrt(var):.4f}
"""
        
        if k is not None:
            theta_hat = self.estimate_gamma_theta(k)
            expected = self.expected_value_gamma(k, theta_hat)
            summary += f"""
Gamma(k={k}) Estimation:
  Estimated theta: {theta_hat:.4f}
  E[T] = k * theta: {expected:.4f}
"""
        else:
            params = self.estimate_gamma_both_params()
            summary += f"""
Gamma Distribution (both parameters):
  Estimated k: {params['k']:.4f}
  Estimated theta: {params['theta']:.4f}
"""
        
        return summary

