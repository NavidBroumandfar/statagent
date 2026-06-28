import numpy as np

from statagent import StatisticalAgent
from statagent.agent import DataExaminer


def test_statistical_agent_rule_based_count_analysis():
    np.random.seed(42)
    data = np.random.negative_binomial(n=10, p=0.3, size=50)

    agent = StatisticalAgent(use_llm=False, verbose=False)
    report = agent.analyze(data=data, goal="understand_distribution")

    assert report.goal == "understand_distribution"
    assert report.data_profile["data_type"] == "discrete_count"
    assert "NegativeBinomialAnalyzer" in report.methods_used
    assert report.get_method_result("NegativeBinomialAnalyzer") is not None


def test_data_examiner_rejects_non_numeric_data():
    try:
        DataExaminer(["low", "medium", "high"])
    except ValueError as exc:
        assert "numeric data" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-numeric data")
