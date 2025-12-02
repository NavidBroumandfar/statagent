# StatAgent

> An intelligent statistical analysis toolkit for probabilistic modeling, hypothesis testing, and Bayesian inference.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**StatAgent** is a comprehensive Python package for advanced statistical analysis, combining classical and Bayesian methods with modern software engineering practices. Built from real-world statistical problems, it provides clean APIs for distribution analysis, parameter estimation, hypothesis testing, and regression modeling.

## Features

### Distribution Analysis
- **Negative Binomial**: Complete PMF/CDF computation, statistical moments, and visualization
- **Survival Functions**: Mixture models with exponential components
- **Simulation**: High-performance random sampling and Monte Carlo methods

### Parameter Estimation
- **Method of Moments**: Classical MoM estimators for Gamma and other distributions
- **Maximum Likelihood**: MLE computation with numerical optimization
- **Bayesian Inference**: Conjugate prior analysis with credible intervals

### Hypothesis Testing
- **Z-tests**: One-sample tests with known variance
- **Confidence Intervals**: Exact and asymptotic intervals
- **P-values**: Left-tailed, right-tailed, and two-tailed tests

### Regression Analysis
- **Polynomial Regression**: OLS fitting with arbitrary degree
- **Ridge Regression**: L2 regularization to prevent overfitting
- **Model Selection**: Automated comparison and validation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/statagent.git
cd statagent

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

#### Negative Binomial Analysis

```python
from statagent import NegativeBinomialAnalyzer

# Analyze meteorite count data
nb = NegativeBinomialAnalyzer(k=63, p=0.32)

# Compute statistics
stats = nb.compute_statistics()
print(f"Mean: {stats['mean']:.2f}")
print(f"Median: {stats['median']}")

# Visualize distribution
nb.plot_pmf(save_path="meteorite_distribution.png")

# Print summary
print(nb.summary())
```

#### Survival Analysis

```python
from statagent import SurvivalMixtureModel

# Model waiting times with mixture of exponentials
model = SurvivalMixtureModel(w1=0.40, rate1=4.0, w2=0.59, rate2=8.0)

# Compute probability
prob = model.compute_probability(2.0, 4.0)
print(f"P(2 ≤ Y ≤ 4) = {prob:.6f}")

# Simulate samples
samples = model.simulate(n=100000)

# Get statistics
stats = model.compute_statistics()
print(f"Mean: {stats['mean']:.4f} hours")

# Visualize
model.plot_pdf(save_path="survival_pdf.png")
```

#### Hypothesis Testing

```python
from statagent import ZTest
import numpy as np

# Test if new system is faster
sample = np.array([1073, 1127, 900, 893, 981, 
                   1050, 922, 1056, 1020, 942])

test = ZTest(sample, mu_0=950.0, sigma=48.3)
result = test.left_tailed_test(alpha=0.05)

print(f"Z-statistic: {result['z_statistic']:.4f}")
print(f"p-value: {result['p_value']:.6f}")
print(f"Reject H0: {result['reject_null']}")

# Get full summary
print(test.summary(test_type="left"))
```

#### Bayesian Estimation

```python
from statagent import BayesianEstimator

# Gamma prior with exponential likelihood
estimator = BayesianEstimator(alpha_prior=48, beta_prior=92)

# Update with data
result = estimator.update_gamma_exponential(
    sample_mean=74.77, 
    n=10, 
    k=3
)

print(f"Posterior mean: {result['bayes_estimate']:.4f}")
print(f"MAP estimate: {result['map_estimate']:.4f}")
print(f"MLE estimate: {result['mle_estimate']:.4f}")

# Get credible interval
interval = estimator.credible_interval(
    result['alpha_post'], 
    result['beta_post']
)
print(f"95% CI: [{interval[0]:.4f}, {interval[1]:.4f}]")
```

#### Polynomial Regression

```python
from statagent import PolynomialRegression
import numpy as np

# Data
X = np.array([-9, 13, 14, -19, 16, 18, 11, 1])
y = np.array([1.72e11, 4.78e11, 1.01e12, 2.71e13, 
              3.83e12, 1.29e13, 8.81e10, 5.80])

# Fit polynomial model
model = PolynomialRegression(degree=10)
model.fit(X, y, lambda_ridge=1e4)

# Make predictions
y_pred = model.predict(X, method='ridge')

# Compare OLS vs Ridge
print(model.summary(X, y))

# Visualize
model.plot_fit(X, y, save_path="regression_fit.png")
```

## Examples

Complete working examples are available in the `examples/` directory:

- **Task 1**: Negative Binomial distribution for meteorite counts
- **Task 2**: Survival analysis with mixture models (owl hearing times)
- **Task 3**: Method of Moments estimation for Gamma distribution
- **Task 4**: Hypothesis testing with Z-tests
- **Task 5**: Polynomial regression with Ridge regularization
- **Task 6**: Bayesian parameter estimation

Run any example:

```bash
python examples/task1_negative_binomial_example.py
```

## Documentation

### Package Structure

```
statagent/
├── distributions/          # Distribution analysis
│   ├── negative_binomial.py
│   └── survival_mixture.py
├── estimation/            # Parameter estimation
│   ├── method_of_moments.py
│   └── bayesian.py
├── inference/             # Hypothesis testing
│   └── hypothesis_tests.py
└── regression/            # Regression models
    └── polynomial.py
```

### API Reference

See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for detailed API documentation.

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=statagent tests/
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black statagent/

# Lint code
flake8 statagent/
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Use Cases

StatAgent is particularly useful for:

- **Scientific Research**: Analyzing experimental data with proper statistical rigor
- **Data Science**: Quick prototyping of statistical models
- **Education**: Learning advanced statistical methods with clean, documented code
- **Production**: Robust statistical analysis in production environments

## Background

This project originated from an Advanced Statistics course assignment covering:

- Discrete probability distributions (Negative Binomial)
- Continuous distributions and mixture models
- Classical estimation theory (Method of Moments, MLE)
- Bayesian inference with conjugate priors
- Hypothesis testing procedures
- Regularized regression methods

The code has been refactored into a professional, reusable library with proper software engineering practices.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built as part of the Advanced Statistics (DLMDSAS01) course
- Inspired by real-world statistical problems in astronomy, biology, and engineering
- Thanks to the scientific Python community (NumPy, SciPy, Matplotlib)

## Contact

**Navid Broumandfar**
- GitHub: [@navidbr](https://github.com/navidbr)
- Email: your.email@example.com

## Star History

If you find this project useful, please consider giving it a star.

---

Made with Python

