# StatAgent Tutorial

A step-by-step guide to using StatAgent for statistical analysis.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/statagent.git
cd statagent

# Install in development mode
pip install -e .
```

## Tutorial 1: Distribution Analysis

### Scenario: Meteorite Count Analysis

You're analyzing the number of meteorites observed per year. Historical data suggests a Negative Binomial distribution fits well.

```python
from statagent import NegativeBinomialAnalyzer

# Create analyzer with known parameters
nb = NegativeBinomialAnalyzer(k=63, p=0.32)

# Get quick statistics
stats = nb.compute_statistics()
print(f"Expected meteorites per year: {stats['mean']:.1f}")
print(f"Most likely range: around {stats['median']}")

# Visualize
nb.plot_pmf(save_path="meteorite_dist.png")
```

**Key Takeaways:**
- High mean (≈133) with large variance indicates high variability
- Median shows the "typical" year
- The distribution is right-skewed

## Tutorial 2: Survival Analysis

### Scenario: Wildlife Observation Times

You're studying how long it takes to observe an owl at night. Times follow a mixture of two processes (close vs far owls).

```python
from statagent import SurvivalMixtureModel

# Model: mixture of two exponentials
model = SurvivalMixtureModel(
    w1=0.40, rate1=4.0,  # Quick observations
    w2=0.59, rate2=8.0   # Slower observations
)

# What's the probability of observing within 2-4 hours?
prob = model.compute_probability(2.0, 4.0)
print(f"P(observe in 2-4 hrs) = {prob:.4f}")

# Simulate a field study
samples = model.simulate(n=1000)
print(f"Average wait: {samples.mean():.2f} hours")
```

**Key Takeaways:**
- Mixture models capture heterogeneous populations
- Can compute exact probabilities or simulate
- Mean wait time combines both processes

## Tutorial 3: Parameter Estimation

### Scenario: Quality Control Data

You have measurements from a manufacturing process and want to estimate parameters.

```python
import numpy as np
from statagent.estimation import MethodOfMoments

# Measurement data
data = np.array([9, 18, 23, 41, 8])

# Estimate parameters assuming Gamma distribution
mom = MethodOfMoments(data)

# If shape k=2 is known from theory
theta_hat = mom.estimate_gamma_theta(k=2)
print(f"Estimated scale: {theta_hat:.2f}")

# Or estimate both parameters
params = mom.estimate_gamma_both_params()
print(f"Estimated k={params['k']:.2f}, theta={params['theta']:.2f}")
```

**Key Takeaways:**
- MoM provides quick, closed-form estimates
- Useful when some parameters are known
- Can estimate all parameters if needed

## Tutorial 4: Hypothesis Testing

### Scenario: Process Improvement

You've implemented a new manufacturing process. Does it really reduce production time?

```python
import numpy as np
from statagent import ZTest

# Old process: mean=950, std=48.3 (known)
# New process: collect sample
new_times = np.array([1073, 1127, 900, 893, 981, 
                      1050, 922, 1056, 1020, 942])

# Test if new process is faster (lower time)
test = ZTest(new_times, mu_0=950, sigma=48.3)
result = test.left_tailed_test(alpha=0.05)

if result['reject_null']:
    print("✓ Significant improvement!")
else:
    print("✗ No significant change")
    
print(f"p-value: {result['p_value']:.4f}")
```

**Key Takeaways:**
- Always state hypotheses clearly (H0 vs H1)
- p-value tells you strength of evidence
- Use appropriate tail based on research question

## Tutorial 5: Regression Analysis

### Scenario: Non-linear Relationship

You have data with a complex non-linear relationship and want to fit a curve.

```python
import numpy as np
from statagent import PolynomialRegression

# Complex data
X = np.array([-9, 13, 14, -19, 16, 18])
y = np.array([1.7e11, 4.8e11, 1.0e12, 2.7e13, 3.8e12, 1.3e13])

# Fit high-degree polynomial
model = PolynomialRegression(degree=10)
model.fit(X, y, lambda_ridge=1e4)

# Compare methods
mse_ols = model.compute_mse(X, y, method='ols')
mse_ridge = model.compute_mse(X, y, method='ridge')

print(f"OLS MSE: {mse_ols:.2e}")
print(f"Ridge MSE: {mse_ridge:.2e}")

# Visualize
model.plot_fit(X, y)
```

**Key Takeaways:**
- Ridge regression prevents overfitting
- High-degree polynomials need regularization
- Always compare training error between methods

## Tutorial 6: Bayesian Inference

### Scenario: Sequential Learning

You have prior knowledge about a parameter and want to update it with new data.

```python
from statagent.estimation import BayesianEstimator

# Prior: based on historical data
estimator = BayesianEstimator(alpha_prior=48, beta_prior=92)

# New data arrives
result = estimator.update_gamma_exponential(
    sample_mean=74.77,
    n=10,
    k=3
)

# Compare estimates
print(f"Prior mean: {48/92:.4f}")
print(f"MLE (data only): {result['mle_estimate']:.4f}")
print(f"Posterior mean: {result['bayes_estimate']:.4f}")

# Quantify uncertainty
ci = estimator.credible_interval(
    result['alpha_post'], 
    result['beta_post']
)
print(f"95% Credible Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
```

**Key Takeaways:**
- Bayesian methods combine prior + data
- Provides natural uncertainty quantification
- Prior pulls estimate away from pure MLE

## Best Practices

### 1. Data Exploration First

Always explore your data before analysis:

```python
import numpy as np

data = np.array([...])
print(f"n: {len(data)}")
print(f"Mean: {data.mean():.2f}")
print(f"Std: {data.std():.2f}")
print(f"Range: [{data.min():.2f}, {data.max():.2f}]")
```

### 2. Validate Assumptions

Check if your data meets model assumptions:

```python
# For Normal-based tests
import matplotlib.pyplot as plt

plt.hist(data, bins=20, density=True)
plt.title("Check for normality")
plt.show()
```

### 3. Report Complete Results

Always report:
- Point estimates AND uncertainty
- Test statistics AND p-values
- Visualizations for interpretation

### 4. Interpret in Context

Statistical significance ≠ practical significance. Always interpret results in the context of your problem.

## Common Workflows

### Complete Analysis Pipeline

```python
# 1. Load data
import numpy as np
data = np.loadtxt("mydata.txt")

# 2. Explore
from statagent.estimation import MethodOfMoments
mom = MethodOfMoments(data)
print(mom.summary())

# 3. Test hypothesis
from statagent import ZTest
test = ZTest(data, mu_0=100, sigma=15)
print(test.summary())

# 4. Visualize
# ... create plots ...

# 5. Report results
# ... write conclusions ...
```

## Troubleshooting

### Issue: "ValueError: Probability must be between 0 and 1"

Check your input parameters:

```python
# Bad
nb = NegativeBinomialAnalyzer(k=63, p=1.5)  # p > 1

# Good
nb = NegativeBinomialAnalyzer(k=63, p=0.32)
```

### Issue: NaN or Inf in results

Usually caused by numerical overflow in polynomial regression. Use Ridge:

```python
model = PolynomialRegression(degree=10)
model.fit(X, y, lambda_ridge=1e4)  # Regularization helps
```

### Issue: Poor model fit

Try different distributions or transformations:

```python
# Log-transform heavy-tailed data
y_log = np.log(y + 1)
model.fit(X, y_log)
```

## Next Steps

- Read the [API Reference](API_REFERENCE.md) for complete method documentation
- Explore the [examples/](../examples/) directory
- Check out advanced topics in the user guide

## Getting Help

- GitHub Issues: Report bugs or request features
- Discussions: Ask questions and share use cases
- Email: Contact the maintainer

---

Happy analyzing! 📊

