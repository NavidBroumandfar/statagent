import pytest

from statagent import BayesianEstimator, MethodOfMoments


def test_method_of_moments_gamma_known_shape():
    estimator = MethodOfMoments([9, 18, 23, 41, 8])

    assert estimator.estimate_gamma_theta(k=2) == pytest.approx(9.9)


def test_method_of_moments_rejects_zero_variance_for_two_parameter_gamma():
    estimator = MethodOfMoments([5, 5, 5])

    with pytest.raises(ValueError, match="variance must be positive"):
        estimator.estimate_gamma_both_params()


def test_bayesian_map_is_not_negative_for_small_shape_posterior():
    estimator = BayesianEstimator(alpha_prior=0.1, beta_prior=1)
    result = estimator.update_gamma_exponential(sample_mean=10, n=1, k=0.1)

    assert result["map_estimate"] == 0.0


def test_bayesian_credible_interval_validates_confidence():
    estimator = BayesianEstimator(alpha_prior=2, beta_prior=1)

    with pytest.raises(ValueError, match="confidence"):
        estimator.credible_interval(alpha_post=3, beta_post=2, confidence=1.5)
