"""
Example: Polynomial Regression with Ridge Regularization
========================================================

This example demonstrates polynomial regression using both OLS and Ridge
regularization to compare their performance on high-degree polynomials.

Problem: Fit a degree-10 polynomial to data with large magnitude variations
and compare OLS vs Ridge performance.
"""

import numpy as np
from statagent import PolynomialRegression

def main():
    print("="*60)
    print("Polynomial Regression: OLS vs Ridge")
    print("="*60)
    
    # Data with large magnitude variations
    data = [
        (-9, 17182267164.34), (13, 478325264808.73),
        (14, 1009292446725.93), (-19, 27109871895252.04),
        (16, 3829242106346.29), (18, 12907538393847.70),
        (11, 88060928800.15), (1, 5.80),
        (-11, 113440455717.35), (4, 3082841.80),
        (-2, 7701.85), (-13, 643674352929.48),
        (-12, 286330096521.11), (-15, 2651768635628.10),
        (7, 901346268.94), (10, 35967434519.66),
        (-16, 4683230003602.72), (-3, 367654.72),
        (-7, 1395957356.69), (3, 152909.93),
        (19, 22533076575710.61), (15, 2165901099171.30)
    ]
    
    X = np.array([d[0] for d in data], dtype=float)
    y = np.array([d[1] for d in data], dtype=float)
    
    print(f"\nData:")
    print(f"  Number of points: {len(X)}")
    print(f"  X range: [{X.min():.0f}, {X.max():.0f}]")
    print(f"  Y range: [{y.min():.2e}, {y.max():.2e}]")
    
    # Fit polynomial model
    degree = 10
    lambda_ridge = 1e4
    
    print(f"\nModel Configuration:")
    print(f"  Polynomial degree: {degree}")
    print(f"  Ridge parameter (λ): {lambda_ridge:.0e}")
    
    model = PolynomialRegression(degree=degree)
    model.fit(X, y, lambda_ridge=lambda_ridge)
    
    # Check for numerical issues
    n_valid_ols = np.sum(np.isfinite(model.coef_ols_))
    n_valid_ridge = np.sum(np.isfinite(model.coef_ridge_))
    
    print(f"\nCoefficient Validity:")
    print(f"  OLS: {n_valid_ols}/{degree+1} finite coefficients")
    print(f"  Ridge: {n_valid_ridge}/{degree+1} finite coefficients")
    
    if n_valid_ols < degree + 1:
        print("  ⚠ Warning: OLS has numerical instability")
    if n_valid_ridge == degree + 1:
        print("  Ridge coefficients are all finite")
    
    # Compute MSE
    mse_ols = model.compute_mse(X, y, method='ols')
    mse_ridge = model.compute_mse(X, y, method='ridge')
    
    print(f"\nMean Squared Error:")
    print(f"  OLS: {mse_ols:.2e}")
    print(f"  Ridge: {mse_ridge:.2e}")
    
    if np.isfinite(mse_ridge) and np.isfinite(mse_ols):
        improvement = (1 - mse_ridge/mse_ols) * 100
        print(f"  Ridge improvement: {improvement:.1f}%")
    
    # Print complete summary
    print("\n" + "="*60)
    print(model.summary(X, y))
    
    # Visualize
    print("\nGenerating visualization...")
    model.plot_fit(
        X, y,
        save_path="figures/polynomial_regression_comparison.png",
        title=f"Polynomial Regression (degree={degree}): OLS vs Ridge"
    )
    print("Figure saved to figures/polynomial_regression_comparison.png")
    
    # Sample predictions
    print("\nSample Predictions:")
    test_x = np.array([0, 5, -10])
    for x_val in test_x:
        y_ols = model.predict(np.array([x_val]), method='ols')[0]
        y_ridge = model.predict(np.array([x_val]), method='ridge')[0]
        print(f"  x = {x_val:3d}: OLS = {y_ols:12.2e}, Ridge = {y_ridge:12.2e}")

if __name__ == "__main__":
    main()

