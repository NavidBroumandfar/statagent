# StatAgent

Experimental Python toolkit for statistical analysis, with a prototype
agent-assisted workflow for choosing and explaining methods.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Project Status

StatAgent is a portfolio and learning project, not a production analytics
platform. The statistical modules are usable for small educational examples,
and the agent layer demonstrates how a data profiler, method selector, and
plain-language interpreter can be wired together.

The default demo path is rule-based and does not require an API key. LLM support
is optional and experimental.

## What It Includes

- Negative Binomial distribution analysis
- Survival-time mixture models with exponential components
- Method of Moments estimation
- Gamma-prior Bayesian estimation for exponential/Gamma likelihoods
- One-sample Z-tests
- Polynomial regression with Ridge regularization
- A prototype `StatisticalAgent` that profiles data, selects a method, runs the
  analysis, and returns a readable report

## Installation

```bash
git clone https://github.com/NavidBroumandfar/statagent.git
cd statagent

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

Optional LLM and notebook dependencies:

```bash
pip install -e ".[llm]"
pip install -e ".[notebooks]"
```

## Quick Examples

### Negative Binomial

```python
from statagent import NegativeBinomialAnalyzer

nb = NegativeBinomialAnalyzer(k=63, p=0.32)
stats = nb.compute_statistics()

print(stats["mean"])
print(nb.summary())
```

### Hypothesis Testing

```python
import numpy as np
from statagent import ZTest

sample = np.array([1073, 1127, 900, 893, 981, 1050, 922, 1056, 1020, 942])
test = ZTest(sample, mu_0=950.0, sigma=48.3)

print(test.left_tailed_test(alpha=0.05))
```

### Rule-Based Agent Prototype

```python
import numpy as np
from statagent import StatisticalAgent

data = np.random.default_rng(42).negative_binomial(n=10, p=0.3, size=100)

agent = StatisticalAgent(use_llm=False, verbose=False)
report = agent.analyze(data, goal="understand_distribution")

print(report.summary())
```

## Polished Demo

Run the portfolio-safe demo:

```bash
python examples/portfolio_demo.py
```

It uses deterministic synthetic count data, runs the rule-based agent path, and
prints a compact report without requiring an API key.

## Optional LLM Mode

The agent can use OpenAI or Ollama for method-selection and interpretation
experiments. This path is intentionally optional because statistical correctness
should come from tested code and explicit assumptions, not from a model response.

```bash
export OPENAI_API_KEY="your-key-here"
pip install -e ".[llm]"
```

```python
from statagent import StatisticalAgent

agent = StatisticalAgent(llm="gpt-4", use_llm=True)
```

## Repository Layout

```text
statagent/
├── distributions/          # Distribution analysis
├── estimation/             # Parameter estimation
├── inference/              # Hypothesis testing
├── regression/             # Regression models
└── agent/                  # Experimental agent layer

tests/                      # Focused pytest coverage
examples/                   # Usage examples and portfolio demo
docs/                       # API and architecture notes
```

## Testing

```bash
pytest
black --check statagent/
flake8 statagent/ --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Limitations

- The agent layer is experimental and should not be treated as an autonomous
  statistical authority.
- Statistical method selection uses simple heuristics unless LLM mode is enabled.
- The toolkit is best suited for educational examples and portfolio review.
- More work is needed before claiming production readiness: stronger tests,
  richer model validation, better data handling, and clearer statistical
  assumptions.

## Background

This project began as an advanced statistics coursework project and was later
refactored into a reusable Python package with an experimental agent interface.

## License

MIT License. See [LICENSE](LICENSE).

## Author

Navid Broumandfar
