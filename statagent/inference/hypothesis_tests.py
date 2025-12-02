"""
Hypothesis testing procedures.

This module implements classical hypothesis tests including Z-tests,
t-tests, and related inference procedures.
"""

import numpy as np
from scipy.stats import norm, t as t_dist
from typing import Dict, Optional


class ZTest:
    """
    Z-test for population mean with known variance.
    
    Tests hypotheses about a population mean when the population
    standard deviation is known.
    
    Parameters
    ----------
    data : array_like
        Sample data
    mu_0 : float
        Null hypothesis value for population mean
    sigma : float
        Known population standard deviation
    
    Attributes
    ----------
    data : np.ndarray
        Sample data
    mu_0 : float
        Null hypothesis mean
    sigma : float
        Population standard deviation
    n : int
        Sample size
    
    Examples
    --------
    >>> sample = np.array([1073, 1127, 900, 893, 981, 1050, 922, 1056, 1020, 942])
    >>> test = ZTest(sample, mu_0=950.0, sigma=48.3)
    >>> result = test.left_tailed_test()
    >>> print(f"p-value: {result['p_value']:.4f}")
    """
    
    def __init__(self, data: np.ndarray, mu_0: float, sigma: float):
        """Initialize the Z-test."""
        self.data = np.asarray(data, dtype=float)
        self.mu_0 = mu_0
        self.sigma = sigma
        self.n = len(self.data)
        
        if self.n == 0:
            raise ValueError("Data array cannot be empty")
        if sigma <= 0:
            raise ValueError("Standard deviation must be positive")
    
    def sample_mean(self) -> float:
        """Compute sample mean."""
        return float(np.mean(self.data))
    
    def standard_error(self) -> float:
        """Compute standard error of the mean."""
        return self.sigma / np.sqrt(self.n)
    
    def z_statistic(self) -> float:
        """
        Compute Z test statistic.
        
        Z = (x_bar - mu_0) / (sigma / sqrt(n))
        
        Returns
        -------
        z : float
            Z test statistic
        """
        x_bar = self.sample_mean()
        se = self.standard_error()
        return (x_bar - self.mu_0) / se
    
    def left_tailed_test(self, alpha: float = 0.05) -> Dict[str, float]:
        """
        Perform left-tailed test (H1: mu < mu_0).
        
        Parameters
        ----------
        alpha : float, optional
            Significance level (default: 0.05)
            
        Returns
        -------
        result : dict
            Dictionary containing:
            - sample_mean: Sample mean
            - z_statistic: Z test statistic
            - p_value: Left-tailed p-value
            - critical_value: Critical value for rejection region
            - reject_null: Whether to reject H0
        """
        z = self.z_statistic()
        p_value = norm.cdf(z)
        critical_value = norm.ppf(alpha)
        reject_null = z < critical_value
        
        return {
            "sample_mean": self.sample_mean(),
            "standard_error": self.standard_error(),
            "z_statistic": float(z),
            "p_value": float(p_value),
            "critical_value": float(critical_value),
            "reject_null": bool(reject_null),
            "alpha": alpha
        }
    
    def right_tailed_test(self, alpha: float = 0.05) -> Dict[str, float]:
        """
        Perform right-tailed test (H1: mu > mu_0).
        
        Parameters
        ----------
        alpha : float, optional
            Significance level (default: 0.05)
            
        Returns
        -------
        result : dict
            Test results dictionary
        """
        z = self.z_statistic()
        p_value = 1 - norm.cdf(z)
        critical_value = norm.ppf(1 - alpha)
        reject_null = z > critical_value
        
        return {
            "sample_mean": self.sample_mean(),
            "standard_error": self.standard_error(),
            "z_statistic": float(z),
            "p_value": float(p_value),
            "critical_value": float(critical_value),
            "reject_null": bool(reject_null),
            "alpha": alpha
        }
    
    def two_tailed_test(self, alpha: float = 0.05) -> Dict[str, float]:
        """
        Perform two-tailed test (H1: mu != mu_0).
        
        Parameters
        ----------
        alpha : float, optional
            Significance level (default: 0.05)
            
        Returns
        -------
        result : dict
            Test results dictionary
        """
        z = self.z_statistic()
        p_value = 2 * (1 - norm.cdf(abs(z)))
        critical_value = norm.ppf(1 - alpha/2)
        reject_null = abs(z) > critical_value
        
        return {
            "sample_mean": self.sample_mean(),
            "standard_error": self.standard_error(),
            "z_statistic": float(z),
            "p_value": float(p_value),
            "critical_value_lower": float(-critical_value),
            "critical_value_upper": float(critical_value),
            "reject_null": bool(reject_null),
            "alpha": alpha
        }
    
    def confidence_interval(self, confidence: float = 0.95) -> tuple:
        """
        Compute confidence interval for population mean.
        
        Parameters
        ----------
        confidence : float, optional
            Confidence level (default: 0.95)
            
        Returns
        -------
        interval : tuple
            (lower, upper) bounds of confidence interval
        """
        alpha = 1 - confidence
        z_critical = norm.ppf(1 - alpha/2)
        x_bar = self.sample_mean()
        se = self.standard_error()
        
        margin = z_critical * se
        return (x_bar - margin, x_bar + margin)
    
    def summary(self, test_type: str = "left", alpha: float = 0.05) -> str:
        """
        Generate a summary of the hypothesis test.
        
        Parameters
        ----------
        test_type : str, optional
            Type of test: "left", "right", or "two" (default: "left")
        alpha : float, optional
            Significance level (default: 0.05)
            
        Returns
        -------
        summary : str
            Formatted summary string
        """
        if test_type == "left":
            result = self.left_tailed_test(alpha)
            h1 = f"H1: μ < {self.mu_0}"
        elif test_type == "right":
            result = self.right_tailed_test(alpha)
            h1 = f"H1: μ > {self.mu_0}"
        else:
            result = self.two_tailed_test(alpha)
            h1 = f"H1: μ ≠ {self.mu_0}"
        
        ci = self.confidence_interval(1 - alpha)
        
        summary = f"""
Z-Test for Population Mean
===========================
Hypotheses:
  H0: μ = {self.mu_0}
  {h1}

Sample:
  n = {self.n}
  Sample mean = {result['sample_mean']:.4f}
  Known σ = {self.sigma}
  Standard error = {result['standard_error']:.4f}

Test Results:
  Z-statistic = {result['z_statistic']:.4f}
  p-value = {result['p_value']:.6f}
  Significance level α = {alpha}
  
Decision:
  {"Reject H0" if result['reject_null'] else "Fail to reject H0"} at α = {alpha}
  
{100*(1-alpha):.0f}% Confidence Interval: [{ci[0]:.2f}, {ci[1]:.2f}]
"""
        
        return summary

