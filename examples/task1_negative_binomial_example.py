"""
Example: Negative Binomial Distribution Analysis
=================================================

This example demonstrates how to use the NegativeBinomialAnalyzer
to analyze discrete count data, such as yearly meteorite observations.

Problem: A space observatory tracks meteorite impacts. The number of 
meteorites observed follows a Negative Binomial distribution with
k=63 successes and p=0.32 success probability.
"""

from statagent import NegativeBinomialAnalyzer

def main():
    print("="*60)
    print("Negative Binomial Analysis: Meteorite Counts")
    print("="*60)
    
    # Initialize analyzer with parameters
    nb = NegativeBinomialAnalyzer(k=63, p=0.32)
    
    # Find significant range
    first, last = nb.find_significant_range(threshold=0.005)
    print(f"\nSignificant Range (P(X=x) >= 0.5%):")
    print(f"  First x: {first}")
    print(f"  Last x: {last}")
    
    # Compute statistics
    stats = nb.compute_statistics()
    print(f"\nStatistical Moments:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Variance: {stats['variance']:.4f}")
    print(f"  Standard Deviation: {stats['std']:.4f}")
    print(f"  Median: {stats['median']}")
    
    # Print complete summary
    print(nb.summary())
    
    # Visualize distribution
    print("\nGenerating visualization...")
    nb.plot_pmf(
        save_path="figures/negative_binomial_meteorites.png",
        title="Negative Binomial PMF: Yearly Meteorite Counts (k=63, p=0.32)"
    )
    print("Figure saved to figures/negative_binomial_meteorites.png")
    
    # Compute specific probabilities
    x_vals, pmf_vals = nb.compute_pmf()
    prob_at_mean = pmf_vals[int(stats['mean'])]
    print(f"\nP(X = {int(stats['mean'])}) ≈ {prob_at_mean:.6f}")

if __name__ == "__main__":
    main()

