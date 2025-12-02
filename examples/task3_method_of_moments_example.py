"""
Example: Method of Moments Estimation
======================================

This example demonstrates parameter estimation using the Method of Moments
for Gamma distributions.

Problem: Estimate the scale parameter theta for a Gamma(k=2, theta) 
distribution from observed data.
"""

import numpy as np
from statagent.estimation import MethodOfMoments

def main():
    print("="*60)
    print("Method of Moments: Gamma Distribution")
    print("="*60)
    
    # Sample data
    data = np.array([9, 18, 23, 41, 8], dtype=float)
    k = 2  # known shape parameter
    
    print(f"\nData: {data}")
    print(f"Known shape parameter: k = {k}")
    
    # Initialize estimator
    mom = MethodOfMoments(data)
    
    # Estimate theta
    theta_hat = mom.estimate_gamma_theta(k=k)
    expected_T = mom.expected_value_gamma(k=k, theta=theta_hat)
    
    print(f"\nEstimation Results:")
    print(f"  θ̂ (theta_hat) = {theta_hat:.4f}")
    print(f"  E[T] = k × θ̂ = {k} × {theta_hat:.4f} = {expected_T:.4f}")
    
    # Print complete summary
    print(mom.summary(k=k))
    
    # Also estimate both parameters
    print("\n" + "="*60)
    print("Estimating both k and θ from data alone:")
    print("="*60)
    
    both_params = mom.estimate_gamma_both_params()
    print(f"\n  k̂ = {both_params['k']:.4f}")
    print(f"  θ̂ = {both_params['theta']:.4f}")
    
    # Verify
    mean_from_params = both_params['k'] * both_params['theta']
    var_from_params = both_params['k'] * both_params['theta']**2
    
    print(f"\nVerification:")
    print(f"  Sample mean: {mom.sample_mean():.4f}")
    print(f"  Mean from k̂ and θ̂: {mean_from_params:.4f}")
    print(f"  Sample variance: {mom.sample_variance():.4f}")
    print(f"  Variance from k̂ and θ̂: {var_from_params:.4f}")

if __name__ == "__main__":
    main()

