# StatAgent Quick Start Guide

Get started with StatAgent in 5 minutes.

## Installation

```bash
cd statagent
pip install -r requirements.txt
pip install -e .
```

## Your First Analysis

### 1. Negative Binomial Distribution

```python
from statagent import NegativeBinomialAnalyzer

# Analyze discrete count data
nb = NegativeBinomialAnalyzer(k=63, p=0.32)
print(nb.summary())
nb.plot_pmf(save_path="output.png")
```

### 2. Hypothesis Testing

```python
from statagent import ZTest
import numpy as np

# Test population mean
data = np.array([1073, 1127, 900, 893, 981])
test = ZTest(data, mu_0=950, sigma=48.3)
result = test.two_tailed_test()

print(f"p-value: {result['p_value']:.4f}")
print(f"Reject H0: {result['reject_null']}")
```

### 3. Bayesian Estimation

```python
from statagent.estimation import BayesianEstimator

# Update prior with data
estimator = BayesianEstimator(alpha_prior=48, beta_prior=92)
result = estimator.update_gamma_exponential(sample_mean=74.77, n=10, k=3)

print(f"Posterior mean: {result['bayes_estimate']:.4f}")
```

## Run Examples

```bash
cd examples
python task1_negative_binomial_example.py
python task4_hypothesis_testing_example.py
python task6_bayesian_estimation_example.py
```

## Use CLI

```bash
# Negative Binomial analysis
python -m statagent.cli negbin -k 63 -p 0.32 --stats --plot

# Z-test
python -m statagent.cli ztest --data 100 105 98 102 99 --mu-0 100 --sigma 5 --summary

# Survival mixture
python -m statagent.cli survival --w1 0.4 --rate1 4.0 --w2 0.59 --rate2 8.0 --stats
```

## Next Steps

1. **Tutorial**: Read [docs/TUTORIAL.md](docs/TUTORIAL.md) for detailed guides
2. **API Reference**: Check [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for all methods
3. **Examples**: Explore [examples/](examples/) directory
4. **Documentation**: See [docs/](docs/) for advanced topics

## Common Tasks

### Load and Analyze Data

```python
import numpy as np
from statagent.estimation import MethodOfMoments

# Load data
data = np.loadtxt("mydata.txt")

# Estimate parameters
mom = MethodOfMoments(data)
params = mom.estimate_gamma_both_params()
print(params)
```

### Survival Analysis

```python
from statagent import SurvivalMixtureModel

model = SurvivalMixtureModel(w1=0.4, rate1=4.0, w2=0.59, rate2=8.0)
samples = model.simulate(n=10000)
model.plot_histogram(save_path="histogram.png")
```

### Polynomial Regression

```python
from statagent import PolynomialRegression
import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 4.8, 9.2, 15.9, 25.1])

model = PolynomialRegression(degree=2)
model.fit(X, y)
model.plot_fit(X, y)
```

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: GitHub Issues for bugs
- **Questions**: GitHub Discussions

Happy analyzing!

