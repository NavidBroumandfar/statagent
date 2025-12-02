# API Reference

Complete API documentation for StatAgent.

## Table of Contents

- [Distributions](#distributions)
  - [NegativeBinomialAnalyzer](#negativebinomialanalyzer)
  - [SurvivalMixtureModel](#survivalmixturemodel)
- [Estimation](#estimation)
  - [MethodOfMoments](#methodofmoments)
  - [BayesianEstimator](#bayesianestimator)
- [Inference](#inference)
  - [ZTest](#ztest)
- [Regression](#regression)
  - [PolynomialRegression](#polynomialregression)

---

## Distributions

### NegativeBinomialAnalyzer

Analyzer for Negative Binomial distributions.

#### Constructor

```python
NegativeBinomialAnalyzer(k: int, p: float)
```

**Parameters:**
- `k` (int): Number of successes (shape parameter)
- `p` (float): Success probability (0 < p < 1)

#### Methods

##### `compute_pmf(x_max=400)`

Compute the probability mass function.

**Returns:** `(x_vals, pmf_vals)` - Arrays of x values and corresponding probabilities

##### `find_significant_range(threshold=0.005)`

Find range where P(X = x) >= threshold.

**Returns:** `(first, last)` - First and last x values meeting threshold

##### `compute_statistics()`

Compute statistical moments.

**Returns:** Dictionary with keys:
- `mean`: Expected value
- `variance`: Variance
- `std`: Standard deviation
- `median`: Median value

##### `plot_pmf(save_path=None, show_stats=True, figsize=(10,5), title=None)`

Plot the probability mass function.

##### `summary()`

Generate text summary of the distribution.

**Returns:** Formatted summary string

---

### SurvivalMixtureModel

Mixture model for survival analysis with exponential components.

#### Constructor

```python
SurvivalMixtureModel(w1: float, rate1: float, w2: float, rate2: float)
```

**Parameters:**
- `w1`, `w2` (float): Component weights (must be non-negative)
- `rate1`, `rate2` (float): Rate parameters (must be positive)

#### Methods

##### `survival_function(y)`

Compute survival function S(y) = P(Y > y).

**Returns:** Array of survival probabilities

##### `pdf(y)`

Compute probability density function.

**Returns:** Array of PDF values

##### `compute_probability(y_lower, y_upper)`

Compute P(y_lower <= Y <= y_upper).

**Returns:** Probability in the interval

##### `compute_statistics()`

Compute theoretical mean and variance.

**Returns:** Dictionary with `mean`, `variance`, `std`

##### `simulate(n=100000, random_state=None)`

Generate random samples from the mixture distribution.

**Returns:** Array of simulated values

##### `plot_pdf(y_max=5.0, save_path=None, show_stats=True, ...)`

Plot the probability density function.

##### `plot_histogram(n=100000, bins=100, save_path=None, ...)`

Plot histogram of simulated samples.

##### `summary()`

Generate text summary of the model.

---

## Estimation

### MethodOfMoments

Method of Moments estimator for various distributions.

#### Constructor

```python
MethodOfMoments(data: array_like)
```

**Parameters:**
- `data`: Observed sample data

#### Methods

##### `sample_mean()`

Compute sample mean.

##### `sample_variance(ddof=1)`

Compute sample variance.

**Parameters:**
- `ddof` (int): Degrees of freedom (default: 1 for unbiased estimate)

##### `estimate_gamma_theta(k)`

Estimate theta parameter for Gamma(k, theta) distribution.

**Parameters:**
- `k` (float): Known shape parameter

**Returns:** Estimated scale parameter

##### `estimate_gamma_both_params()`

Estimate both k and theta for Gamma distribution.

**Returns:** Dictionary with keys `k` and `theta`

##### `expected_value_gamma(k, theta=None)`

Compute expected value for Gamma distribution.

##### `summary(k=None)`

Generate estimation summary.

---

### BayesianEstimator

Bayesian parameter estimator with Gamma prior.

#### Constructor

```python
BayesianEstimator(alpha_prior: float, beta_prior: float)
```

**Parameters:**
- `alpha_prior` (float): Prior shape parameter
- `beta_prior` (float): Prior rate parameter

#### Methods

##### `update_gamma_exponential(sample_mean, n, k=1)`

Update Gamma prior with exponential/Gamma likelihood.

**Parameters:**
- `sample_mean` (float): Sample mean of observations
- `n` (int): Sample size
- `k` (float): Shape parameter (default: 1 for exponential)

**Returns:** Dictionary with:
- `alpha_post`: Posterior shape
- `beta_post`: Posterior rate
- `bayes_estimate`: Posterior mean
- `map_estimate`: Posterior mode
- `mle_estimate`: Maximum likelihood estimate

##### `posterior_variance(alpha_post, beta_post)`

Compute posterior variance.

##### `credible_interval(alpha_post, beta_post, confidence=0.95)`

Compute Bayesian credible interval.

**Returns:** `(lower, upper)` bounds

##### `summary(sample_mean, n, k=1, show_interval=True)`

Generate estimation summary.

---

## Inference

### ZTest

Z-test for population mean with known variance.

#### Constructor

```python
ZTest(data: array_like, mu_0: float, sigma: float)
```

**Parameters:**
- `data`: Sample data
- `mu_0` (float): Null hypothesis value for population mean
- `sigma` (float): Known population standard deviation

#### Methods

##### `sample_mean()`

Compute sample mean.

##### `standard_error()`

Compute standard error of the mean.

##### `z_statistic()`

Compute Z test statistic.

**Returns:** Z test statistic

##### `left_tailed_test(alpha=0.05)`

Perform left-tailed test (H1: mu < mu_0).

**Returns:** Dictionary with test results including:
- `sample_mean`
- `z_statistic`
- `p_value`
- `critical_value`
- `reject_null`

##### `right_tailed_test(alpha=0.05)`

Perform right-tailed test (H1: mu > mu_0).

##### `two_tailed_test(alpha=0.05)`

Perform two-tailed test (H1: mu != mu_0).

##### `confidence_interval(confidence=0.95)`

Compute confidence interval for population mean.

**Returns:** `(lower, upper)` bounds

##### `summary(test_type='left', alpha=0.05)`

Generate test summary.

**Parameters:**
- `test_type` (str): "left", "right", or "two"

---

## Regression

### PolynomialRegression

Polynomial regression with OLS and Ridge regularization.

#### Constructor

```python
PolynomialRegression(degree: int = 10)
```

**Parameters:**
- `degree` (int): Degree of polynomial

#### Methods

##### `fit(x, y, lambda_ridge=1e4)`

Fit polynomial regression using OLS and Ridge.

**Parameters:**
- `x` (array_like): Input features (1D)
- `y` (array_like): Target values (1D)
- `lambda_ridge` (float): Ridge regularization parameter

**Returns:** self (fitted model)

##### `predict(x, method='ridge')`

Make predictions using fitted model.

**Parameters:**
- `x` (array_like): Input features
- `method` (str): 'ols' or 'ridge'

**Returns:** Predicted values

##### `plot_fit(x, y, x_range=None, n_points=500, ...)`

Plot the fitted polynomial curves.

##### `compute_mse(x, y, method='ridge')`

Compute mean squared error.

**Returns:** MSE value

##### `summary(x, y)`

Generate model summary with comparison of OLS vs Ridge.

---

## Common Patterns

### Basic Workflow

```python
# 1. Import
from statagent import NegativeBinomialAnalyzer

# 2. Initialize
analyzer = NegativeBinomialAnalyzer(k=63, p=0.32)

# 3. Compute
stats = analyzer.compute_statistics()

# 4. Visualize
analyzer.plot_pmf(save_path="output.png")

# 5. Summarize
print(analyzer.summary())
```

### Error Handling

All classes validate inputs and raise `ValueError` for invalid parameters:

```python
try:
    nb = NegativeBinomialAnalyzer(k=-1, p=0.5)  # Invalid k
except ValueError as e:
    print(f"Error: {e}")
```

### Customization

Most plotting methods accept customization parameters:

```python
model.plot_pdf(
    y_max=10.0,
    save_path="custom.png",
    figsize=(12, 8),
    title="My Custom Title",
    show_stats=True
)
```

## Type Hints

All methods include type hints for better IDE support:

```python
def compute_statistics(self) -> Dict[str, float]:
    ...
```

## NumPy Integration

All numerical methods work seamlessly with NumPy arrays:

```python
import numpy as np

x = np.linspace(0, 10, 100)
y = model.pdf(x)  # Vectorized computation
```

