"""
Example: Hypothesis Testing with Z-Test
========================================

This example demonstrates hypothesis testing using Z-tests when the
population standard deviation is known.

Problem: A factory claims their new system produces items faster (lower
completion time) than the old system with mean 950. Test this claim.
"""

import numpy as np
from statagent import ZTest

def main():
    print("="*60)
    print("Z-Test: Production System Comparison")
    print("="*60)
    
    # Hypotheses
    print("\nHypotheses:")
    print("  H₀: μ = 950 (no improvement)")
    print("  H₁: μ < 950 (new system is faster)")
    
    # Parameters
    mu_0 = 950.0      # historical mean
    sigma = 48.3      # known population standard deviation
    
    # Sample from new system
    sample = np.array([1073, 1127, 900, 893, 981,
                      1050, 922, 1056, 1020, 942])
    
    print(f"\nData:")
    print(f"  Historical mean (μ₀): {mu_0}")
    print(f"  Known σ: {sigma}")
    print(f"  Sample size: {len(sample)}")
    print(f"  Sample: {sample}")
    
    # Initialize test
    test = ZTest(sample, mu_0=mu_0, sigma=sigma)
    
    # Perform left-tailed test
    result = test.left_tailed_test(alpha=0.05)
    
    print(f"\nTest Statistics:")
    print(f"  Sample mean: {result['sample_mean']:.4f}")
    print(f"  Standard error: {result['standard_error']:.4f}")
    print(f"  Z-statistic: {result['z_statistic']:.4f}")
    print(f"  p-value: {result['p_value']:.6f}")
    print(f"  Critical value: {result['critical_value']:.4f}")
    
    print(f"\nDecision (α = 0.05):")
    if result['reject_null']:
        print("  Reject H₀")
        print("  The new system is NOT significantly faster")
    else:
        print("  Fail to reject H₀")
        print("  Insufficient evidence that new system is faster")
    
    # Compute confidence interval
    ci = test.confidence_interval(confidence=0.95)
    print(f"\n95% Confidence Interval: [{ci[0]:.2f}, {ci[1]:.2f}]")
    
    # Print complete summary
    print("\n" + "="*60)
    print(test.summary(test_type="left", alpha=0.05))
    
    # Try two-tailed test
    print("\n" + "="*60)
    print("Alternative: Two-Tailed Test")
    print("="*60)
    
    result_two = test.two_tailed_test(alpha=0.05)
    print(f"\nH₁: μ ≠ {mu_0}")
    print(f"  Z-statistic: {result_two['z_statistic']:.4f}")
    print(f"  p-value: {result_two['p_value']:.6f}")
    print(f"  Reject H₀: {result_two['reject_null']}")

if __name__ == "__main__":
    main()

