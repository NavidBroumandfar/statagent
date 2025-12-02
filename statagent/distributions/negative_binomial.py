"""
Negative Binomial distribution analysis.

This module provides tools for analyzing discrete count data using the
Negative Binomial distribution, including PMF computation, statistical
moments, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from typing import Dict, Tuple, Optional


class NegativeBinomialAnalyzer:
    """
    Analyzer for Negative Binomial distributions.
    
    The Negative Binomial distribution models the number of failures before
    achieving a specified number of successes in a sequence of independent
    Bernoulli trials.
    
    Parameters
    ----------
    k : int
        Number of successes (shape parameter)
    p : float
        Success probability (must be between 0 and 1)
    
    Attributes
    ----------
    k : int
        Number of successes
    p : float
        Success probability
    q : float
        Failure probability (1 - p)
    
    Examples
    --------
    >>> nb = NegativeBinomialAnalyzer(k=63, p=0.32)
    >>> stats = nb.compute_statistics()
    >>> print(f"Mean: {stats['mean']:.2f}")
    >>> nb.plot_pmf(save_path="distribution.png")
    """
    
    def __init__(self, k: int, p: float):
        """Initialize the Negative Binomial analyzer."""
        if not 0 < p < 1:
            raise ValueError("Probability p must be between 0 and 1")
        if k <= 0:
            raise ValueError("Number of successes k must be positive")
            
        self.k = k
        self.p = p
        self.q = 1 - p
        
    def compute_pmf(self, x_max: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the probability mass function.
        
        Parameters
        ----------
        x_max : int, optional
            Maximum value to compute PMF for (default: 400)
            
        Returns
        -------
        x_vals : np.ndarray
            Array of x values
        pmf_vals : np.ndarray
            Corresponding PMF values
        """
        x_vals = np.arange(0, x_max + 1)
        pmf_vals = nbinom.pmf(x_vals, self.k, self.p)
        return x_vals, pmf_vals
    
    def find_significant_range(self, threshold: float = 0.005) -> Tuple[int, int]:
        """
        Find range where P(X = x) >= threshold.
        
        Parameters
        ----------
        threshold : float, optional
            Minimum probability threshold (default: 0.005 = 0.5%)
            
        Returns
        -------
        first : int
            First x value meeting threshold
        last : int
            Last x value meeting threshold
        """
        x_vals, pmf_vals = self.compute_pmf()
        mask = pmf_vals >= threshold
        indices = x_vals[mask]
        
        return int(indices.min()), int(indices.max())
    
    def compute_statistics(self) -> Dict[str, float]:
        """
        Compute statistical moments of the distribution.
        
        Returns
        -------
        stats : dict
            Dictionary containing:
            - mean: Expected value
            - variance: Variance
            - std: Standard deviation
            - median: Median value
        """
        mean = nbinom.mean(self.k, self.p)
        var = nbinom.var(self.k, self.p)
        std = np.sqrt(var)
        
        # Compute median via CDF
        median = self._compute_median()
        
        return {
            "mean": float(mean),
            "variance": float(var),
            "std": float(std),
            "median": int(median)
        }
    
    def _compute_median(self, x_max: int = 2000) -> int:
        """Compute median by finding smallest m where P(X <= m) >= 0.5."""
        cdf = 0.0
        for xi in range(0, x_max + 1):
            cdf += nbinom.pmf(xi, self.k, self.p)
            if cdf >= 0.5:
                return xi
        return x_max
    
    def plot_pmf(
        self,
        save_path: Optional[str] = None,
        show_stats: bool = True,
        figsize: Tuple[int, int] = (10, 5),
        title: Optional[str] = None
    ) -> None:
        """
        Plot the probability mass function.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure (if None, figure is not saved)
        show_stats : bool, optional
            Whether to show mean and median lines (default: True)
        figsize : tuple, optional
            Figure size (width, height) in inches
        title : str, optional
            Custom plot title (if None, default title is used)
        """
        stats = self.compute_statistics()
        _, last = self.find_significant_range()
        
        x_plot = np.arange(0, last + 1)
        pmf_plot = nbinom.pmf(x_plot, self.k, self.p)
        
        plt.figure(figsize=figsize)
        plt.bar(x_plot, pmf_plot, edgecolor="black", alpha=0.7, label="PMF")
        
        if show_stats:
            plt.axvline(
                stats["mean"],
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean ≈ {stats['mean']:.2f}"
            )
            plt.axvline(
                stats["median"],
                color="green",
                linestyle=":",
                linewidth=2,
                label=f"Median = {stats['median']}"
            )
        
        if title is None:
            title = f"Negative Binomial PMF (k={self.k}, p={self.p})"
        
        plt.title(title)
        plt.xlabel("Number of events (x)")
        plt.ylabel("Probability P(X = x)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def summary(self) -> str:
        """
        Generate a text summary of the distribution.
        
        Returns
        -------
        summary : str
            Formatted summary string
        """
        stats = self.compute_statistics()
        first, last = self.find_significant_range()
        
        summary = f"""
Negative Binomial Distribution Analysis
========================================
Parameters:
  k (successes): {self.k}
  p (success probability): {self.p}
  q (failure probability): {self.q}

Statistics:
  Mean: {stats['mean']:.4f}
  Variance: {stats['variance']:.4f}
  Standard Deviation: {stats['std']:.4f}
  Median: {stats['median']}

Significant Range (P(X=x) >= 0.5%):
  First x: {first}
  Last x: {last}
  Range: [{first}, {last}]
"""
        return summary

