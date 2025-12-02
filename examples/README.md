# Examples

This directory contains complete working examples demonstrating all features of StatAgent.

## Running Examples

Each example is a standalone Python script that can be run directly:

```bash
python task1_negative_binomial_example.py
```

Make sure you have installed the package and dependencies:

```bash
pip install -r ../requirements.txt
pip install -e ..
```

## Example Files

### Task 1: Negative Binomial Distribution
**File:** `task1_negative_binomial_example.py`

Demonstrates analysis of discrete count data using the Negative Binomial distribution:
- Computing PMF and statistical moments
- Finding significant probability ranges
- Visualizing distributions with mean and median markers

**Use case:** Analyzing rare event counts like meteorite observations, customer arrivals, or defect counts.

### Task 2: Survival Analysis with Mixture Models
**File:** `task2_survival_mixture_example.py`

Shows how to model survival times using mixture distributions:
- Mixture of exponential distributions
- Survival function computation
- Monte Carlo simulation for empirical statistics
- PDF and histogram visualization

**Use case:** Modeling waiting times, failure rates, or duration analysis with heterogeneous populations.

### Task 3: Method of Moments Estimation
**File:** `task3_method_of_moments_example.py`

Illustrates parameter estimation using the Method of Moments:
- Estimating scale parameter with known shape
- Estimating both shape and scale parameters
- Verifying estimates against sample statistics

**Use case:** Quick parameter estimation for Gamma, Normal, and other parametric distributions.

### Task 4: Hypothesis Testing
**File:** `task4_hypothesis_testing_example.py`

Demonstrates classical hypothesis testing with Z-tests:
- Left-tailed, right-tailed, and two-tailed tests
- P-value computation and interpretation
- Confidence interval construction
- Statistical decision making

**Use case:** Comparing means, quality control, A/B testing, scientific experiments.

### Task 5: Polynomial Regression
**File:** `task5_polynomial_regression_example.py`

Shows advanced regression with regularization:
- High-degree polynomial fitting
- OLS vs Ridge regression comparison
- Handling numerical instability
- Visualization of fitted curves

**Use case:** Non-linear relationship modeling, overfitting prevention, curve fitting.

### Task 6: Bayesian Estimation
**File:** `task6_bayesian_estimation_example.py`

Demonstrates Bayesian inference with conjugate priors:
- Gamma prior with exponential/Gamma likelihood
- Posterior updating
- Comparing Bayes, MAP, and MLE estimators
- Credible interval computation

**Use case:** Incorporating prior knowledge, sequential updating, uncertainty quantification.

## Output

Each example generates:
- Console output with detailed statistics
- Visualizations saved to the `figures/` directory (automatically created)

## Customization

All examples can be easily customized by changing parameters in the `main()` function:

```python
# Modify parameters
nb = NegativeBinomialAnalyzer(k=100, p=0.5)  # Different parameters

# Change visualization settings
nb.plot_pmf(
    save_path="my_custom_plot.png",
    figsize=(12, 6),
    title="My Custom Title"
)
```

## Interactive Notebooks

For interactive exploration, Jupyter notebook versions are available in the project root.

