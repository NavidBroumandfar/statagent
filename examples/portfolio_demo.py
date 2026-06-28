"""
Portfolio-safe StatAgent demo.

This demo uses deterministic synthetic count data and the rule-based agent path,
so it runs without an OpenAI key or local LLM server.
"""

import numpy as np

from statagent import NegativeBinomialAnalyzer, StatisticalAgent


def main() -> None:
    rng = np.random.default_rng(42)
    count_data = rng.negative_binomial(n=10, p=0.3, size=100)

    agent = StatisticalAgent(use_llm=False, verbose=False)
    report = agent.analyze(count_data, goal="understand_distribution")

    print("StatAgent Portfolio Demo")
    print("========================")
    print(report.summary())

    nb_result = report.get_method_result("NegativeBinomialAnalyzer")
    if nb_result is not None:
        stats = nb_result["output"]["statistics"]
        print("Selected Distribution")
        print("---------------------")
        print(f"Mean: {stats['mean']:.2f}")
        print(f"Variance: {stats['variance']:.2f}")
        print(f"Median: {stats['median']}")

    baseline = NegativeBinomialAnalyzer(k=10, p=0.3)
    baseline_stats = baseline.compute_statistics()
    print("Baseline Reference")
    print("------------------")
    print(f"Expected mean for k=10, p=0.3: {baseline_stats['mean']:.2f}")


if __name__ == "__main__":
    main()
