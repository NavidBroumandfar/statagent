"""
Example: Bayesian Parameter Estimation
=======================================

This example demonstrates Bayesian inference using conjugate priors,
comparing posterior estimates with MLE.

Problem: Estimate a rate parameter theta using Gamma prior and
exponential likelihood, comparing Bayes, MAP, and MLE estimators.
"""

import numpy as np
from statagent.estimation import BayesianEstimator

def main():
    print("="*60)
    print("Bayesian Estimation: Gamma Prior + Exponential Likelihood")
    print("="*60)
    
    # Prior parameters
    alpha_prior = 48
    beta_prior = 92
    
    print(f"\nPrior Distribution:")
    print(f"  θ ~ Gamma(α = {alpha_prior}, β = {beta_prior})")
    print(f"  Prior mean: {alpha_prior/beta_prior:.4f}")
    print(f"  Prior variance: {alpha_prior/beta_prior**2:.6f}")
    
    # Data
    n = 10
    sample_mean = 74.77
    k = 3  # shape parameter for likelihood
    
    print(f"\nLikelihood:")
    print(f"  X_i ~ Gamma(k = {k}, θ)")
    print(f"  Sample size: n = {n}")
    print(f"  Sample mean: {sample_mean:.2f}")
    
    # Initialize Bayesian estimator
    estimator = BayesianEstimator(
        alpha_prior=alpha_prior,
        beta_prior=beta_prior
    )
    
    # Update with data
    result = estimator.update_gamma_exponential(
        sample_mean=sample_mean,
        n=n,
        k=k
    )
    
    print(f"\nPosterior Distribution:")
    print(f"  θ | data ~ Gamma(α = {result['alpha_post']:.0f}, β = {result['beta_post']:.2f})")
    
    # Posterior variance
    post_var = estimator.posterior_variance(
        result['alpha_post'],
        result['beta_post']
    )
    print(f"  Posterior mean: {result['bayes_estimate']:.6f}")
    print(f"  Posterior variance: {post_var:.8f}")
    
    # Point estimates
    print(f"\nPoint Estimates:")
    print(f"  Bayes estimate (posterior mean): {result['bayes_estimate']:.6f}")
    print(f"  MAP estimate (posterior mode):   {result['map_estimate']:.6f}")
    print(f"  MLE estimate (likelihood only):   {result['mle_estimate']:.6f}")
    
    # Compare estimates
    print(f"\nComparison:")
    diff_bayes_mle = abs(result['bayes_estimate'] - result['mle_estimate'])
    diff_map_mle = abs(result['map_estimate'] - result['mle_estimate'])
    
    print(f"  |Bayes - MLE|: {diff_bayes_mle:.6f}")
    print(f"  |MAP - MLE|:   {diff_map_mle:.6f}")
    
    # Credible interval
    ci_95 = estimator.credible_interval(
        result['alpha_post'],
        result['beta_post'],
        confidence=0.95
    )
    
    ci_99 = estimator.credible_interval(
        result['alpha_post'],
        result['beta_post'],
        confidence=0.99
    )
    
    print(f"\nCredible Intervals:")
    print(f"  95% CI: [{ci_95[0]:.6f}, {ci_95[1]:.6f}]")
    print(f"  99% CI: [{ci_99[0]:.6f}, {ci_99[1]:.6f}]")
    
    # Check if MLE is in credible interval
    mle_in_ci = ci_95[0] <= result['mle_estimate'] <= ci_95[1]
    print(f"\n  MLE in 95% CI: {mle_in_ci}")
    
    # Print complete summary
    print("\n" + "="*60)
    print(estimator.summary(
        sample_mean=sample_mean,
        n=n,
        k=k,
        show_interval=True
    ))
    
    # Interpretation
    print("="*60)
    print("Interpretation:")
    print("="*60)
    print("""
The Bayesian estimate incorporates prior knowledge with the data,
resulting in a compromise between the prior mean and the MLE.

- If the prior is weak (large variance), Bayes ≈ MLE
- If the prior is strong (small variance), Bayes pulls toward prior mean
- MAP is similar to Bayes but uses the mode instead of mean

In this case, the posterior is influenced by both the prior and the
observed data, providing a regularized estimate with quantified uncertainty.
""")

if __name__ == "__main__":
    main()

