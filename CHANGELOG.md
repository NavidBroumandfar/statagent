# Changelog

All notable changes to StatAgent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-01

### Added
- Initial release of StatAgent
- **Distribution Analysis**
  - `NegativeBinomialAnalyzer` for discrete count data
  - `SurvivalMixtureModel` for survival analysis with exponential mixtures
- **Parameter Estimation**
  - `MethodOfMoments` for classical MoM estimation
  - `BayesianEstimator` for conjugate prior analysis
- **Hypothesis Testing**
  - `ZTest` for population mean testing with known variance
  - Support for left-tailed, right-tailed, and two-tailed tests
  - Confidence interval computation
- **Regression Analysis**
  - `PolynomialRegression` with OLS and Ridge regularization
  - Automatic handling of numerical instability
  - Visualization of fitted curves
- **Visualization**
  - High-quality plotting for all analyses
  - Automatic figure generation with sensible defaults
  - Customizable plot parameters
- **Documentation**
  - Comprehensive README with quick start guide
  - API reference with all methods
  - Tutorial with 6 complete examples
  - Contributing guidelines
- **Examples**
  - 6 detailed example scripts covering all features
  - Real-world scenarios from astronomy, biology, manufacturing
- **CLI Interface**
  - Command-line tools for common analyses
  - Support for negative binomial, survival, and hypothesis testing
- **Testing**
  - Package structure ready for pytest
  - Example test patterns in documentation

### Package Structure
- Modular architecture with clear separation of concerns
- Type hints for better IDE support
- NumPy-style docstrings throughout
- Clean public API via `__init__.py` exports

### Dependencies
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- pandas >= 1.3.0

## [Unreleased]

### Planned
- Additional distribution analyzers (Poisson, Weibull, etc.)
- T-tests and non-parametric tests
- MCMC sampling for complex posteriors
- Time series analysis module
- Extended regression methods (Lasso, ElasticNet)
- Interactive Jupyter widgets
- More comprehensive test suite
- Performance optimizations for large datasets

---

## Version History

- **v0.1.0** (2025-12-01): Initial release with core functionality

