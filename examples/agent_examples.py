"""
Autonomous Statistical Analysis Examples

This file demonstrates the StatisticalAgent's ability to autonomously:
1. Examine data characteristics
2. Select appropriate statistical methods
3. Execute analyses
4. Interpret results
5. Provide recommendations

The agent operates with minimal human intervention.
"""

import numpy as np
import os

from statagent import StatisticalAgent

def example_count_data():
    """Example: Analyzing count data"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Count Data Analysis")
    print("="*80)
    
    # Simulated count data (e.g., events per observation period)
    np.random.seed(42)
    data = np.random.negative_binomial(n=10, p=0.3, size=100)
    
    print(f"\nData: {len(data)} observations of count data")
    print(f"Sample: {data[:10]}...")
    
    # Create agent (uses rule-based mode if no API key)
    agent = StatisticalAgent(
        llm="gpt-4",
        verbose=True,
        use_llm=False  # Set to True if OpenAI API key is available
    )
    
    # Agent analyzes autonomously
    report = agent.analyze(
        data=data,
        goal="understand_distribution"
    )
    
    # Display summary
    print("\n" + report.summary())
    
    # Access specific results
    print("\nAccessing Specific Results:")
    nb_result = report.get_method_result("NegativeBinomialAnalyzer")
    if nb_result:
        stats = nb_result['output']['statistics']
        print(f"  Distribution Mean: {stats['mean']:.2f}")
        print(f"  Distribution Median: {stats['median']}")
        print(f"  Standard Deviation: {stats['std']:.2f}")


def example_hypothesis_testing():
    """Example: Autonomous hypothesis testing"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Hypothesis Testing")
    print("="*80)
    
    # Simulated data: system response times
    np.random.seed(42)
    data = np.random.normal(loc=1050, scale=50, size=30)
    
    print(f"\nData: {len(data)} response time measurements")
    print(f"Sample mean: {np.mean(data):.2f}")
    
    # Create agent
    agent = StatisticalAgent(verbose=True, use_llm=False)
    
    # Agent autonomously selects and executes hypothesis test
    report = agent.analyze(
        data=data,
        goal="test_hypothesis",
        hypothesis="mean > 1000"
    )
    
    print("\n" + report.summary())
    
    # Check hypothesis test results
    ztest_result = report.get_method_result("ZTest")
    if ztest_result:
        reject = ztest_result['output'].get('reject_null', False)
        p_value = ztest_result['output'].get('p_value', 1.0)
        print(f"\nHypothesis Test Result:")
        print(f"  Reject null hypothesis: {reject}")
        print(f"  P-value: {p_value:.6f}")


def example_continuous_data():
    """Example: Comprehensive analysis of continuous data"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Comprehensive Analysis")
    print("="*80)
    
    # Simulated continuous data
    np.random.seed(42)
    data = np.random.exponential(scale=2.5, size=50)
    
    print(f"\nData: {len(data)} continuous observations")
    print(f"Sample: {data[:5]}...")
    
    # Create agent
    agent = StatisticalAgent(verbose=True, use_llm=False)
    
    # Comprehensive analysis
    report = agent.analyze(
        data=data,
        goal="comprehensive_statistical_analysis"
    )
    
    print("\n" + report.summary())
    
    # View recommendations
    print("\nAgent Recommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")


def example_with_llm():
    """Example: Using LLM for enhanced reasoning (requires API key)"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Enhanced Analysis with LLM")
    print("="*80)
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n[INFO] OpenAI API key not found.")
        print("       Set OPENAI_API_KEY environment variable for LLM features.")
        print("       Example: export OPENAI_API_KEY='your-key-here'")
        print("\n       Running in rule-based mode instead.")
        use_llm = False
    else:
        print("\n[OK] OpenAI API key found. Using GPT-4 for reasoning.")
        use_llm = True
    
    # Mixed data (challenging to classify)
    np.random.seed(42)
    data = np.concatenate([
        np.random.poisson(lam=5, size=30),
        np.random.poisson(lam=15, size=20)
    ])
    
    print(f"\nData: {len(data)} observations (mixture of two processes)")
    
    # Create agent with LLM
    agent = StatisticalAgent(
        llm="gpt-4",
        verbose=True,
        use_llm=use_llm
    )
    
    # Agent reasons about the data
    report = agent.analyze(
        data=data,
        goal="identify_distribution_and_estimate_parameters"
    )
    
    print("\n" + report.summary())
    
    if use_llm:
        # View agent's reasoning process
        print("\nAgent's Reasoning Process:")
        reasoning = agent.explain_reasoning()
        print(reasoning[:1000] + "..." if len(reasoning) > 1000 else reasoning)


def example_quick_analysis():
    """Example: Quick analysis mode"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Quick Analysis Mode")
    print("="*80)
    
    # Random data
    np.random.seed(42)
    data = np.random.gamma(shape=2, scale=3, size=40)
    
    print(f"\nData: {len(data)} observations")
    
    # Quick analysis with minimal output
    agent = StatisticalAgent(verbose=False, use_llm=False)
    summary = agent.quick_analyze(data)
    
    print("\nQuick Analysis:")
    print(summary)


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("STATISTICAL AGENT - AUTONOMOUS ANALYSIS EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate autonomous data analysis:")
    print("- Data examination")
    print("- Method selection")
    print("- Analysis execution")
    print("- Result interpretation")
    print("- Recommendation generation")
    
    # Run examples
    example_count_data()
    example_hypothesis_testing()
    example_continuous_data()
    example_quick_analysis()
    
    # LLM example (optional)
    print("\n" + "="*80)
    print("Enhanced LLM Example (Optional)")
    print("="*80)
    example_with_llm()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. Agent examines data characteristics automatically")
    print("  2. Selects appropriate methods based on data type and goal")
    print("  3. Executes analyses with minimal configuration")
    print("  4. Interprets results in plain language")
    print("  5. Provides actionable recommendations")
    print("\nTry the agent with your own data!")


if __name__ == "__main__":
    main()

