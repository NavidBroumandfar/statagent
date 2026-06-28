import numpy as np
import pytest

from statagent import ZTest


def test_ztest_left_tailed_result():
    sample = np.array([1073, 1127, 900, 893, 981, 1050, 922, 1056, 1020, 942])
    test = ZTest(sample, mu_0=950.0, sigma=48.3)
    result = test.left_tailed_test(alpha=0.05)

    assert result["sample_mean"] == pytest.approx(996.4)
    assert result["z_statistic"] == pytest.approx(3.0379, abs=1e-4)
    assert result["reject_null"] is False


def test_ztest_rejects_empty_data():
    with pytest.raises(ValueError, match="cannot be empty"):
        ZTest([], mu_0=0, sigma=1)
