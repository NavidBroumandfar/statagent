#!/usr/bin/env python3
"""
StatAgent Interactive Demo
==========================

This script demonstrates all major features of StatAgent.
Run this after installing to see what the package can do.

Author: Navid Broumandfar
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for demo

from statagent import (
    NegativeBinomialAnalyzer,
    SurvivalMixtureModel,
    ZTest,
    PolynomialRegression
)
from statagent.estimation import MethodOfMoments, BayesianEstimator


def print_header(title):
    """Print section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_subheader(title):
    """Print subsection header."""
    print(f"\n{title}")
    print("-"*70)


def demo_distribution_analysis():
    """Demo: Negative Binomial distribution analysis."""
    print_header("DEMO 1: Distribution Analysis")
    print("\nScenario: Analyzing yearly meteorite observation counts")
    
    # Create analyzer
    nb = NegativeBinomialAnalyzer(k=63, p=0.32)
    
    # Compute statistics
    stats = nb.compute_statistics()
    first, last = nb.find_significant_range()
    
    print_subheader("Results")
    print(f"Expected meteorites per year: {stats['mean']:.1f}")
    print(f"Standard deviation: {stats['std']:.1f}")
    print(f"Median (typical year): {stats['median']}")
    print(f"Likely range: {first} to {last} meteorites")
    
    # Generate plot
    nb.plot_pmf(save_path='figures/demo_negbin.png')
    print(f"\nVisualization saved: figures/demo_negbin.png")


def demo_hypothesis_testing():
    """Demo: Statistical hypothesis testing."""
    print_header("DEMO 2: Hypothesis Testing")
    print("\nScenario: Testing if new manufacturing process is faster")
    
    # Data
    old_mean = 950  # historical process time
    known_std = 48.3
    new_times = np.array([1073, 1127, 900, 893, 981, 
                          1050, 922, 1056, 1020, 942])
    
    print(f"\nOld process mean: {old_mean} units")
    print(f"New process sample: {new_times[:5]}... (n={len(new_times)})")
    
    # Run test
    test = ZTest(new_times, mu_0=old_mean, sigma=known_std)
    result = test.two_tailed_test(alpha=0.05)
    
    print_subheader("Results")
    print(f"New process mean: {result['sample_mean']:.1f} units")
    print(f"Z-statistic: {result['z_statistic']:.4f}")
    print(f"P-value: {result['p_value']:.6f}")
    
    if result['reject_null']:
        print(f"\nConclusion: Processes are DIFFERENT (p < 0.05)")
    else:
        print(f"\nConclusion: No significant difference (p > 0.05)")


def demo_bayesian_estimation():
    """Demo: Bayesian parameter estimation."""
    print_header("DEMO 3: Bayesian Estimation")
    print("\nScenario: Updating beliefs about a parameter with new data")
    
    # Prior
    alpha_prior = 48
    beta_prior = 92
    prior_mean = alpha_prior / beta_prior
    
    print(f"\nPrior belief: mean = {prior_mean:.4f}")
    
    # New data
    sample_mean = 74.77
    n = 10
    
    print(f"New data: {n} observations, mean = {sample_mean:.2f}")
    
    # Update
    estimator = BayesianEstimator(alpha_prior=alpha_prior, beta_prior=beta_prior)
    result = estimator.update_gamma_exponential(sample_mean=sample_mean, n=n, k=3)
    
    print_subheader("Results")
    print(f"Prior mean:           {prior_mean:.4f}")
    print(f"Posterior mean (Bayes): {result['bayes_estimate']:.4f}")
    print(f"MLE (data only):      {result['mle_estimate']:.4f}")
    
    print(f"\nInterpretation:")
    print(f"  Prior pulled the estimate from {result['mle_estimate']:.4f} (MLE)")
    print(f"  to {result['bayes_estimate']:.4f} (Bayes), incorporating")
    print(f"  prior knowledge with observed data.")


def demo_survival_analysis():
    """Demo: Survival analysis with mixture models."""
    print_header("DEMO 4: Survival Analysis")
    print("\nScenario: Modeling waiting time to observe wildlife")
    
    # Create model
    model = SurvivalMixtureModel(
        w1=0.40, rate1=4.0,  # Quick observations
        w2=0.59, rate2=8.0   # Longer waits
    )
    
    # Statistics
    stats = model.compute_statistics()
    prob_2_4 = model.compute_probability(2.0, 4.0)
    
    print_subheader("Results")
    print(f"Mean waiting time: {stats['mean']:.3f} hours ({stats['mean']*60:.1f} min)")
    print(f"Standard deviation: {stats['std']:.3f} hours")
    print(f"Probability of observing in 2-4 hours: {prob_2_4:.6f}")
    
    # Simulate
    samples = model.simulate(n=10000, random_state=42)
    print(f"\nSimulation (10,000 observations):")
    print(f"  Empirical mean: {np.mean(samples):.3f} hours")
    print(f"  25th percentile: {np.percentile(samples, 25)*60:.1f} min")
    print(f"  50th percentile: {np.percentile(samples, 50)*60:.1f} min")
    print(f"  75th percentile: {np.percentile(samples, 75)*60:.1f} min")
    
    # Generate plot
    model.plot_pdf(save_path='figures/demo_survival.png')
    print(f"\nVisualization saved: figures/demo_survival.png")


def demo_regression():
    """Demo: Polynomial regression with regularization."""
    print_header("DEMO 5: Polynomial Regression")
    print("\nScenario: Fitting complex relationship with regularization")
    
    # Data
    X = np.array([-9, 13, 14, -19, 16, 18, 11, 1, -11, 4])
    y = np.array([1.72e11, 4.78e11, 1.01e12, 2.71e13, 3.83e12, 
                  1.29e13, 8.81e10, 5.80, 1.13e11, 3.08e6])
    
    print(f"Data points: {len(X)}")
    print(f"X range: [{X.min()}, {X.max()}]")
    print(f"Y range: [{y.min():.2e}, {y.max():.2e}]")
    
    # Fit model
    model = PolynomialRegression(degree=10)
    model.fit(X, y, lambda_ridge=1e4)
    
    # Compare
    mse_ols = model.compute_mse(X, y, method='ols')
    mse_ridge = model.compute_mse(X, y, method='ridge')
    
    print_subheader("Results")
    print(f"OLS Mean Squared Error:   {mse_ols:.2e}")
    print(f"Ridge Mean Squared Error: {mse_ridge:.2e}")
    
    if np.isfinite(mse_ridge) and np.isfinite(mse_ols):
        improvement = (1 - mse_ridge/mse_ols) * 100
        print(f"Ridge improvement: {improvement:.1f}%")
    
    print(f"\nConclusion: Ridge regularization prevents overfitting")
    
    # Generate plot
    model.plot_fit(X, y, save_path='figures/demo_regression.png')
    print(f"Visualization saved: figures/demo_regression.png")


def demo_method_of_moments():
    """Demo: Parameter estimation via Method of Moments."""
    print_header("DEMO 6: Method of Moments")
    print("\nScenario: Estimating parameters from sample data")
    
    # Data
    data = np.array([9, 18, 23, 41, 8])
    k_known = 2
    
    print(f"Sample data: {data}")
    print(f"Sample size: {len(data)}")
    print(f"Known shape parameter: k = {k_known}")
    
    # Estimate
    mom = MethodOfMoments(data)
    theta_hat = mom.estimate_gamma_theta(k=k_known)
    
    print_subheader("Results")
    print(f"Sample mean: {mom.sample_mean():.2f}")
    print(f"Sample variance: {mom.sample_variance():.2f}")
    print(f"\nEstimated scale parameter: θ = {theta_hat:.2f}")
    print(f"Expected value: E[X] = k×θ = {k_known}×{theta_hat:.2f} = {k_known*theta_hat:.2f}")


def main():
    """Run complete demo."""
    print("\n" + "="*70)
    print("  STATAGENT - INTERACTIVE DEMO")
    print("  Statistical Analysis Toolkit")
    print("  Author: Navid Broumandfar")
    print("="*70)
    
    print("\nThis demo showcases all major features of StatAgent.")
    print("Visual outputs will be saved to the figures/ directory.")
    
    try:
        # Run all demos
        demo_distribution_analysis()
        demo_hypothesis_testing()
        demo_bayesian_estimation()
        demo_survival_analysis()
        demo_regression()
        demo_method_of_moments()
        
        # Summary
        print_header("DEMO COMPLETE")
        print("\nAll features demonstrated successfully!")
        print("\nGenerated visualizations:")
        print("  - figures/demo_negbin.png")
        print("  - figures/demo_survival.png")
        print("  - figures/demo_regression.png")
        
        print("\nNext steps:")
        print("  1. View the generated figures")
        print("  2. Explore examples/ directory for detailed code")
        print("  3. Read docs/TUTORIAL.md for step-by-step guides")
        print("  4. Check docs/API_REFERENCE.md for complete API")
        print("  5. Try using StatAgent with your own data")
        
        print("\nFor help:")
        print("  - Read README.md")
        print("  - Check documentation in docs/")
        print("  - Open an issue on GitHub")
        
        print("\n" + "="*70)
        print("  Thank you for trying StatAgent!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

