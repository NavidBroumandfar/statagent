"""
Survival function and mixture model analysis.

This module implements survival analysis using mixture models of exponential
distributions, with support for PDF/CDF computation, simulation, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from typing import Dict, Tuple, Optional


class SurvivalMixtureModel:
    """
    Mixture model for survival analysis with exponential components.
    
    Models survival times using a mixture of two exponential distributions:
    S(y) = w1 * exp(-rate1 * y) + w2 * exp(-rate2 * y)
    
    Parameters
    ----------
    w1 : float
        Weight of first component
    rate1 : float
        Rate parameter of first exponential
    w2 : float
        Weight of second component
    rate2 : float
        Rate parameter of second exponential
    
    Attributes
    ----------
    w1, w2 : float
        Component weights
    rate1, rate2 : float
        Rate parameters
    normalization : float
        Normalization constant for PDF
    
    Examples
    --------
    >>> model = SurvivalMixtureModel(w1=0.40, rate1=4.0, w2=0.59, rate2=8.0)
    >>> stats = model.compute_statistics()
    >>> samples = model.simulate(n=100000)
    >>> model.plot_pdf(save_path="survival_pdf.png")
    """
    
    def __init__(self, w1: float, rate1: float, w2: float, rate2: float):
        """Initialize the survival mixture model."""
        if w1 < 0 or w2 < 0:
            raise ValueError("Weights must be non-negative")
        if rate1 <= 0 or rate2 <= 0:
            raise ValueError("Rate parameters must be positive")
            
        self.w1 = w1
        self.rate1 = rate1
        self.w2 = w2
        self.rate2 = rate2
        
        # Compute normalization constant
        self.normalization = self._compute_normalization()
        
        # Normalized weights for mixture
        self._mix_w1 = w1 / (w1 + w2)
        self._mix_w2 = w2 / (w1 + w2)
    
    def survival_function(self, y: np.ndarray) -> np.ndarray:
        """
        Compute survival function S(y) = P(Y > y).
        
        Parameters
        ----------
        y : array_like
            Time values
            
        Returns
        -------
        s : np.ndarray
            Survival probabilities
        """
        y = np.asarray(y)
        return self.w1 * np.exp(-self.rate1 * y) + self.w2 * np.exp(-self.rate2 * y)
    
    def _pdf_unnormalized(self, y: np.ndarray) -> np.ndarray:
        """Compute unnormalized PDF (derivative of -S(y))."""
        y = np.asarray(y)
        return (self.w1 * self.rate1 * np.exp(-self.rate1 * y) +
                self.w2 * self.rate2 * np.exp(-self.rate2 * y))
    
    def _compute_normalization(self) -> float:
        """Compute normalization constant by integrating unnormalized PDF."""
        integral, _ = quad(lambda t: self._pdf_unnormalized(t), 0, np.inf)
        return integral
    
    def pdf(self, y: np.ndarray) -> np.ndarray:
        """
        Compute probability density function.
        
        Parameters
        ----------
        y : array_like
            Time values
            
        Returns
        -------
        f : np.ndarray
            PDF values
        """
        return self._pdf_unnormalized(y) / self.normalization
    
    def compute_probability(self, y_lower: float, y_upper: float) -> float:
        """
        Compute P(y_lower <= Y <= y_upper).
        
        Parameters
        ----------
        y_lower : float
            Lower bound
        y_upper : float
            Upper bound
            
        Returns
        -------
        prob : float
            Probability in the interval
        """
        s_lower = self.survival_function(y_lower)
        s_upper = self.survival_function(y_upper)
        return (s_lower - s_upper) / self.normalization
    
    def compute_statistics(self) -> Dict[str, float]:
        """
        Compute theoretical mean and variance.
        
        Returns
        -------
        stats : dict
            Dictionary containing:
            - mean: Expected value
            - variance: Variance
            - std: Standard deviation
        """
        # Using mixture representation
        EY = self._mix_w1 * (1 / self.rate1) + self._mix_w2 * (1 / self.rate2)
        EY2 = self._mix_w1 * (2 / self.rate1**2) + self._mix_w2 * (2 / self.rate2**2)
        VarY = EY2 - EY**2
        
        return {
            "mean": float(EY),
            "variance": float(VarY),
            "std": float(np.sqrt(VarY))
        }
    
    def simulate(self, n: int = 100000, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from the mixture distribution.
        
        Parameters
        ----------
        n : int, optional
            Number of samples to generate (default: 100000)
        random_state : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        samples : np.ndarray
            Array of simulated values
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Mixture sampling
        u = np.random.rand(n)
        samples = np.empty(n)
        
        # First component
        mask1 = u < self._mix_w1
        n1 = mask1.sum()
        u1 = np.random.rand(n1)
        samples[mask1] = -np.log(1 - u1) / self.rate1
        
        # Second component
        n2 = n - n1
        u2 = np.random.rand(n2)
        samples[~mask1] = -np.log(1 - u2) / self.rate2
        
        return samples
    
    def plot_pdf(
        self,
        y_max: float = 5.0,
        save_path: Optional[str] = None,
        show_stats: bool = True,
        figsize: Tuple[int, int] = (10, 5),
        title: Optional[str] = None
    ) -> None:
        """
        Plot the probability density function.
        
        Parameters
        ----------
        y_max : float, optional
            Maximum y value to plot (default: 5.0)
        save_path : str, optional
            Path to save the figure
        show_stats : bool, optional
            Whether to show mean and median lines
        figsize : tuple, optional
            Figure size (width, height) in inches
        title : str, optional
            Custom plot title
        """
        stats = self.compute_statistics()
        
        y_vals = np.linspace(0, y_max, 500)
        pdf_vals = self.pdf(y_vals)
        
        plt.figure(figsize=figsize)
        plt.plot(y_vals, pdf_vals, linewidth=2, label="PDF")
        
        if show_stats:
            plt.axvline(
                stats["mean"],
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean ≈ {stats['mean']:.3f}"
            )
        
        if title is None:
            title = "Survival Mixture Model PDF"
        
        plt.title(title)
        plt.xlabel("Time (y)")
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_histogram(
        self,
        n: int = 100000,
        bins: int = 100,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 5),
        title: Optional[str] = None
    ) -> None:
        """
        Plot histogram of simulated samples.
        
        Parameters
        ----------
        n : int, optional
            Number of samples to simulate
        bins : int, optional
            Number of histogram bins
        save_path : str, optional
            Path to save the figure
        figsize : tuple, optional
            Figure size (width, height) in inches
        title : str, optional
            Custom plot title
        """
        samples = self.simulate(n)
        
        plt.figure(figsize=figsize)
        plt.hist(samples, bins=bins, density=True, edgecolor="black", alpha=0.7)
        
        if title is None:
            title = f"Histogram of Simulated Samples (n={n:,})"
        
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def summary(self) -> str:
        """
        Generate a text summary of the model.
        
        Returns
        -------
        summary : str
            Formatted summary string
        """
        stats = self.compute_statistics()
        
        summary = f"""
Survival Mixture Model Analysis
================================
Model: S(y) = w1*exp(-r1*y) + w2*exp(-r2*y)

Parameters:
  Component 1: w1 = {self.w1}, rate1 = {self.rate1}
  Component 2: w2 = {self.w2}, rate2 = {self.rate2}
  Normalization: {self.normalization:.6f}

Mixture Weights (normalized):
  Weight 1: {self._mix_w1:.4f}
  Weight 2: {self._mix_w2:.4f}

Statistics:
  Mean: {stats['mean']:.6f}
  Variance: {stats['variance']:.6f}
  Standard Deviation: {stats['std']:.6f}
"""
        return summary

