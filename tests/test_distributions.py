import numpy as np
import pytest

from statagent import NegativeBinomialAnalyzer, SurvivalMixtureModel


def test_negative_binomial_statistics_match_scipy_parameterization():
    analyzer = NegativeBinomialAnalyzer(k=10, p=0.4)
    stats = analyzer.compute_statistics()

    assert stats["mean"] == pytest.approx(15.0)
    assert stats["variance"] == pytest.approx(37.5)
    assert stats["std"] == pytest.approx(np.sqrt(37.5))


def test_negative_binomial_significant_range_reports_empty_threshold():
    analyzer = NegativeBinomialAnalyzer(k=10, p=0.5)

    with pytest.raises(ValueError, match="No PMF values"):
        analyzer.find_significant_range(threshold=1.0)


def test_survival_mixture_rejects_zero_total_weight():
    with pytest.raises(ValueError, match="At least one mixture weight"):
        SurvivalMixtureModel(w1=0, rate1=1, w2=0, rate2=2)


def test_survival_function_is_normalized_when_weights_are_not():
    model = SurvivalMixtureModel(w1=2, rate1=1, w2=2, rate2=2)

    assert model.survival_function(0) == pytest.approx(1.0)
    assert model.compute_probability(0, 100) == pytest.approx(1.0, abs=1e-8)


def test_survival_simulation_is_reproducible():
    model = SurvivalMixtureModel(w1=0.4, rate1=4, w2=0.6, rate2=8)

    first = model.simulate(n=10, random_state=7)
    second = model.simulate(n=10, random_state=7)

    assert np.allclose(first, second)
