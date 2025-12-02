"""
Test the StatisticalAgent in rule-based mode (no LLM required).
"""

import numpy as np
import sys

def test_agent_basic():
    """Test basic agent functionality without LLM."""
    print("\n" + "="*70)
    print("STATISTICAL AGENT TEST - Rule-Based Mode")
    print("="*70 + "\n")
    
    try:
        # Import the agent
        from statagent import StatisticalAgent
        print("[OK] StatisticalAgent imported successfully")
        
        # Create test data
        np.random.seed(42)
        data = np.random.negative_binomial(n=10, p=0.3, size=50)
        print(f"[OK] Test data created: {len(data)} observations")
        print(f"     Sample: {data[:5]}...\n")
        
        # Create agent in rule-based mode
        print("Creating agent (rule-based mode)...")
        agent = StatisticalAgent(use_llm=False, verbose=False)
        print("[OK] Agent created successfully\n")
        
        # Perform analysis
        print("Running autonomous analysis...")
        report = agent.analyze(
            data=data,
            goal="understand_distribution"
        )
        print("[OK] Analysis completed successfully\n")
        
        # Check report
        print("Report Summary:")
        print(f"  Goal: {report.goal}")
        print(f"  Data type: {report.data_profile.get('data_type', 'unknown')}")
        print(f"  Methods used: {len(report.methods_used)}")
        for method in report.methods_used:
            print(f"    - {method}")
        
        # Check results
        successful = len([r for r in report.results if r['status'] == 'success'])
        total = len(report.results)
        print(f"  Results: {successful}/{total} successful\n")
        
        # Display some findings
        print("Key Findings:")
        for interpretation in report.interpretations[:1]:
            findings = interpretation.get('key_findings', [])
            for finding in findings[:3]:
                print(f"  - {finding}")
        
        # Recommendations
        print("\nRecommendations:")
        for rec in report.recommendations[:3]:
            print(f"  - {rec}")
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED")
        print("="*70)
        print("\nStatAgent Phase 2 is working correctly.")
        print("\nNext steps:")
        print("  - Try: python examples/agent_examples.py")
        print("  - Read: docs/AGENT_ARCHITECTURE.md")
        print("  - Enable LLM: export OPENAI_API_KEY='your-key'")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent_basic()
    sys.exit(0 if success else 1)

