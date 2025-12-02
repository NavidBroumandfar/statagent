# Contributing to StatAgent

Thank you for your interest in contributing to StatAgent! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and constructive. We welcome contributions from everyone.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

1. **Clear title**: Describe the problem concisely
2. **Steps to reproduce**: Minimal code to reproduce the bug
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: Python version, OS, package versions

Example:

```
Title: NegativeBinomialAnalyzer raises error with large k

Steps:
1. nb = NegativeBinomialAnalyzer(k=1000, p=0.01)
2. nb.compute_statistics()

Expected: Returns statistics dictionary
Actual: Raises MemoryError

Environment: Python 3.9, macOS 13, numpy 1.24
```

### Suggesting Features

Create an issue with:

1. **Use case**: Why is this feature needed?
2. **Proposed API**: How should it work?
3. **Alternatives**: What alternatives exist?

### Pull Requests

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/my-feature`
3. **Make changes**: Follow coding standards
4. **Add tests**: Ensure new code is tested
5. **Update docs**: Document new features
6. **Commit**: Use clear commit messages
7. **Push**: `git push origin feature/my-feature`
8. **Open PR**: Describe your changes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/statagent.git
cd statagent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black statagent/

# Lint code
flake8 statagent/
```

## Coding Standards

### Style Guide

Follow PEP 8 with these specifics:

- **Line length**: 88 characters (Black default)
- **Imports**: Group stdlib, third-party, local
- **Docstrings**: NumPy style
- **Type hints**: Use for public APIs

Example:

```python
from typing import Dict, Optional

import numpy as np
from scipy import stats


class MyAnalyzer:
    """
    Brief description.
    
    Longer description explaining the purpose and usage.
    
    Parameters
    ----------
    param1 : type
        Description
    param2 : type, optional
        Description (default: value)
    
    Examples
    --------
    >>> analyzer = MyAnalyzer(param1=10)
    >>> result = analyzer.compute()
    """
    
    def compute(self) -> Dict[str, float]:
        """
        Compute analysis results.
        
        Returns
        -------
        results : dict
            Dictionary with analysis results
        """
        pass
```

### Testing

Write tests for all new functionality:

```python
# tests/test_distributions.py
import pytest
import numpy as np
from statagent import NegativeBinomialAnalyzer


def test_negative_binomial_mean():
    """Test that mean calculation is correct."""
    nb = NegativeBinomialAnalyzer(k=10, p=0.5)
    stats = nb.compute_statistics()
    expected_mean = 10 * (1 - 0.5) / 0.5
    assert abs(stats['mean'] - expected_mean) < 1e-6


def test_invalid_probability():
    """Test that invalid probability raises error."""
    with pytest.raises(ValueError):
        NegativeBinomialAnalyzer(k=10, p=1.5)
```

Run tests:

```bash
# All tests
pytest

# With coverage
pytest --cov=statagent --cov-report=html

# Specific file
pytest tests/test_distributions.py

# Specific test
pytest tests/test_distributions.py::test_negative_binomial_mean
```

### Documentation

#### Docstrings

Use NumPy style docstrings:

```python
def my_function(x: np.ndarray, option: bool = True) -> float:
    """
    Brief one-line description.
    
    More detailed description if needed. Can span multiple
    lines and include mathematical notation using LaTeX.
    
    The function computes:
    .. math:: f(x) = \\sum_{i=1}^n x_i^2
    
    Parameters
    ----------
    x : np.ndarray
        Input array of shape (n,)
    option : bool, optional
        Whether to apply option (default: True)
    
    Returns
    -------
    result : float
        Computed result
    
    Raises
    ------
    ValueError
        If x is empty
    
    See Also
    --------
    related_function : Related functionality
    
    Notes
    -----
    Additional implementation notes or mathematical background.
    
    References
    ----------
    .. [1] Author, "Paper Title", Journal, Year.
    
    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> my_function(x)
    14.0
    """
    if len(x) == 0:
        raise ValueError("x cannot be empty")
    return float(np.sum(x**2))
```

#### README Updates

Update README.md if you:
- Add new features
- Change public APIs
- Update installation process

#### API Reference

Update `docs/API_REFERENCE.md` for new classes/methods.

## Project Structure

```
statagent/
├── statagent/           # Main package
│   ├── __init__.py
│   ├── distributions/
│   ├── estimation/
│   ├── inference/
│   └── regression/
├── tests/               # Test suite
│   ├── test_distributions.py
│   ├── test_estimation.py
│   └── ...
├── examples/            # Example scripts
├── docs/                # Documentation
├── README.md
├── setup.py
└── requirements.txt
```

## Commit Messages

Use clear, descriptive commit messages:

```
Add Ridge regression to PolynomialRegression

- Implement Ridge regularization
- Add lambda parameter to fit()
- Update tests and documentation
- Closes #42
```

Format:
- **First line**: Brief summary (50 chars)
- **Body**: Detailed description (wrap at 72 chars)
- **Footer**: Reference issues

## Release Process

(For maintainers)

1. Update version in `setup.py` and `__init__.py`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.2.0`
4. Push tag: `git push origin v0.2.0`
5. Create GitHub release
6. Build and upload to PyPI:
   ```bash
   python setup.py sdist bdist_wheel
   twine upload dist/*
   ```

## Questions?

Feel free to:
- Open an issue for discussion
- Email the maintainer
- Join our community discussions

Thank you for contributing! 🎉

