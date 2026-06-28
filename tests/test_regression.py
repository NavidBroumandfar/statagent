import numpy as np
import pytest

from statagent import PolynomialRegression


def test_polynomial_regression_predicts_training_shape():
    x = np.array([0, 1, 2, 3, 4], dtype=float)
    y = np.array([1, 2, 5, 10, 17], dtype=float)

    model = PolynomialRegression(degree=2).fit(x, y, lambda_ridge=0.1)
    predictions = model.predict(x, method="ridge")

    assert predictions.shape == y.shape
    assert np.all(np.isfinite(predictions))


def test_polynomial_regression_marks_underdetermined_ols():
    x = np.array([1, 2, 3], dtype=float)
    y = np.array([1, 4, 9], dtype=float)

    model = PolynomialRegression(degree=10).fit(x, y)

    assert model.ols_rank_deficient_ is True


def test_polynomial_regression_validates_inputs():
    model = PolynomialRegression(degree=2)

    with pytest.raises(ValueError, match="same length"):
        model.fit([1, 2], [1])
