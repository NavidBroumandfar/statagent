"""
Example: Survival Analysis with Mixture Models
===============================================

This example demonstrates survival analysis using a mixture of exponential
distributions to model waiting times.

Problem: A researcher observes how long it takes to hear an owl at night.
The waiting time follows a mixture of two exponential distributions.
"""

import numpy as np
from statagent import SurvivalMixtureModel

def main():
    print("="*60)
    print("Survival Mixture Model: Owl Hearing Times")
    print("="*60)
    
    # Initialize model with parameters
    model = SurvivalMixtureModel(
        w1=0.40,    # weight of first component
        rate1=4.0,  # rate of first exponential
        w2=0.59,    # weight of second component
        rate2=8.0   # rate of second exponential
    )
    
    # Compute probability in interval
    prob_2_4 = model.compute_probability(2.0, 4.0)
    print(f"\nProbability Analysis:")
    print(f"  P(2 ≤ Y ≤ 4 hours) = {prob_2_4:.6e}")
    
    # Compute theoretical statistics
    stats = model.compute_statistics()
    print(f"\nTheoretical Statistics:")
    print(f"  Mean: {stats['mean']:.6f} hours ({stats['mean']*60:.3f} minutes)")
    print(f"  Variance: {stats['variance']:.6f} hours²")
    print(f"  Std Dev: {stats['std']:.6f} hours")
    
    # Simulate samples
    print(f"\nRunning simulation (n=100,000)...")
    samples = model.simulate(n=100_000, random_state=42)
    
    # Compute empirical statistics
    mean_sim = np.mean(samples)
    var_sim = np.var(samples)
    q1, median, q3 = np.quantile(samples, [0.25, 0.5, 0.75])
    
    print(f"\nEmpirical Statistics:")
    print(f"  Mean: {mean_sim:.6f} hours ({mean_sim*60:.3f} minutes)")
    print(f"  Variance: {var_sim:.6f} hours²")
    print(f"  Q1: {q1*60:.3f} min")
    print(f"  Median: {median*60:.3f} min")
    print(f"  Q3: {q3*60:.3f} min")
    
    # Print complete summary
    print(model.summary())
    
    # Visualize PDF
    print("\nGenerating PDF visualization...")
    model.plot_pdf(
        y_max=5.0,
        save_path="figures/survival_mixture_pdf.png",
        title="Survival Mixture PDF: Owl Waiting Time"
    )
    print("PDF figure saved to figures/survival_mixture_pdf.png")
    
    # Visualize histogram
    print("Generating histogram...")
    model.plot_histogram(
        n=100_000,
        bins=100,
        save_path="figures/survival_mixture_histogram.png",
        title="Histogram: Owl Waiting Time (100,000 samples)"
    )
    print("Histogram saved to figures/survival_mixture_histogram.png")

if __name__ == "__main__":
    main()

