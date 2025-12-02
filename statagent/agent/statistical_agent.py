"""
Statistical Agent - Main interface for autonomous statistical analysis.

This module provides the main StatisticalAgent class that orchestrates
all components for intelligent, autonomous statistical analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any
import warnings
import time

from statagent.agent.data_examiner import DataExaminer
from statagent.agent.reasoning_engine import ReasoningEngine
from statagent.agent.orchestrator import Orchestrator
from statagent.agent.interpreter import Interpreter


class AnalysisReport:
    """
    Container for analysis results and insights.
    
    Attributes
    ----------
    goal : str
        Analysis goal
    data_profile : dict
        Data characteristics
    methods_used : list
        Methods executed
    results : list
        All execution results
    interpretations : list
        All interpretations
    reasoning_history : list
        LLM reasoning steps
    recommendations : list
        Suggested next steps
    """
    
    def __init__(self, goal: str, data_profile: Dict, results: List[Dict],
                 interpretations: List[Dict], reasoning_history: List[Dict],
                 recommendations: List[str]):
        """Initialize analysis report."""
        self.goal = goal
        self.data_profile = data_profile
        self.results = results
        self.interpretations = interpretations
        self.reasoning_history = reasoning_history
        self.recommendations = recommendations
        self.methods_used = [r['method'] for r in results if r['status'] == 'success']
    
    def summary(self) -> str:
        """Get summary of analysis."""
        successful = len([r for r in self.results if r['status'] == 'success'])
        total = len(self.results)
        
        summary = f"""
Statistical Analysis Summary
============================
Goal: {self.goal}
Data: {self.data_profile.get('n', 0)} observations ({self.data_profile.get('data_type', 'unknown')})
Methods executed: {successful}/{total} successful
        
Key Methods Used:
"""
        for method in self.methods_used:
            summary += f"  • {method}\n"
        
        summary += "\nTop Recommendations:\n"
        for rec in self.recommendations[:3]:
            summary += f"  • {rec}\n"
        
        return summary
    
    def get_successful_results(self) -> List[Dict]:
        """Get only successful results."""
        return [r for r in self.results if r['status'] == 'success']
    
    def get_method_result(self, method_name: str) -> Optional[Dict]:
        """Get result for specific method."""
        for result in self.results:
            if result['method'] == method_name and result['status'] == 'success':
                return result
        return None


class StatisticalAgent:
    """
    Autonomous Statistical Analysis Agent.
    
    The StatisticalAgent examines data, reasons about appropriate methods,
    executes analyses, and interprets results autonomously.
    
    Parameters
    ----------
    llm : str, optional
        LLM to use for reasoning: "gpt-4", "gpt-3.5-turbo", "ollama/llama3"
        Default: "gpt-4"
    api_key : str, optional
        OpenAI API key (if None, reads from OPENAI_API_KEY env var)
    verbose : bool, optional
        Whether to print analysis progress (default: True)
    temperature : float, optional
        LLM temperature (default: 0.1 for consistency)
    use_llm : bool, optional
        Whether to use LLM for reasoning (default: True)
        If False, uses rule-based fallbacks only
    
    Attributes
    ----------
    verbose : bool
        Verbosity flag
    use_llm : bool
        Whether LLM is being used
    analysis_history : list
        History of all analyses performed
    
    Examples
    --------
    >>> agent = StatisticalAgent(llm="gpt-4", verbose=True)
    >>> report = agent.analyze(data, goal="understand_distribution")
    >>> print(report.summary())
    
    >>> # Without LLM (rule-based only)
    >>> agent = StatisticalAgent(use_llm=False, verbose=True)
    >>> report = agent.analyze(data)
    """
    
    def __init__(
        self,
        llm: str = "gpt-4",
        api_key: Optional[str] = None,
        verbose: bool = True,
        temperature: float = 0.1,
        use_llm: bool = True,
    ):
        """Initialize the Statistical Agent."""
        self.verbose = verbose
        self.use_llm = use_llm
        self.analysis_history = []
        
        # Initialize components
        if use_llm:
            try:
                self.reasoning_engine = ReasoningEngine(
                    llm=llm,
                    api_key=api_key,
                    verbose=verbose,
                    temperature=temperature,
                )
                if self.verbose:
                    print(f"Initialized with {llm}")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to initialize LLM: {e}")
                    print("   Falling back to rule-based analysis")
                self.use_llm = False
                self.reasoning_engine = None
        else:
            self.reasoning_engine = None
            if self.verbose:
                print("Initialized in rule-based mode (no LLM)")
    
    def analyze(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame, list],
        goal: str = "comprehensive_statistical_analysis",
        hypothesis: Optional[str] = None,
        **kwargs
    ) -> AnalysisReport:
        """
        Perform autonomous statistical analysis.
        
        This is the main entry point for the agent. It will:
        1. Examine data characteristics
        2. Reason about appropriate methods
        3. Execute analyses
        4. Interpret results
        5. Suggest next steps
        
        Parameters
        ----------
        data : array_like, Series, or DataFrame
            Data to analyze
        goal : str, optional
            Analysis goal/objective. Examples:
            - "understand_distribution"
            - "test_hypothesis"
            - "estimate_parameters"
            - "comprehensive_statistical_analysis"
        hypothesis : str, optional
            Specific hypothesis to test (e.g., "mean > 100")
        **kwargs
            Additional parameters for specific analyses
        
        Returns
        -------
        report : AnalysisReport
            Complete analysis report with results and recommendations
        """
        if self.verbose:
            print("\n" + "="*70)
            print("STATISTICAL AGENT - AUTONOMOUS ANALYSIS")
            print("="*70 + "\n")
            start_time = time.time()
        
        # Step 1: Examine Data
        if self.verbose:
            print("STEP 1: Examining data...")
        
        examiner = DataExaminer(data, verbose=self.verbose)
        data_profile = examiner.examine()
        
        if self.verbose:
            print("\n" + examiner.summary())
        
        # Step 2: Reason about methods
        if self.verbose:
            print("STEP 2: Reasoning about appropriate methods...")
        
        if self.use_llm and self.reasoning_engine:
            # LLM-powered reasoning
            data_analysis = self.reasoning_engine.analyze_data(data_profile, goal)
            method_selection = self.reasoning_engine.select_methods(
                data_profile, goal, data_analysis.get('analysis')
            )
        else:
            # Rule-based fallback
            data_analysis = None
            method_selection = self._rule_based_method_selection(
                data_profile, goal, hypothesis
            )
        
        # Step 3: Execute analyses
        if self.verbose:
            print("\nSTEP 3: Executing analyses...")
        
        orchestrator = Orchestrator(
            data=examiner.data,
            verbose=self.verbose
        )
        
        # Execute each selected method
        results = []
        for method_spec in method_selection.get('methods', []):
            method_name = method_spec.get('name', '')
            parameters = method_spec.get('parameters', {})
            
            if self.verbose:
                print(f"\n   Executing {method_name}...")
                print(f"   Rationale: {method_spec.get('rationale', 'N/A')}")
            
            result = orchestrator.execute_method(method_name, parameters)
            results.append(result)
        
        # Step 4: Interpret results
        if self.verbose:
            print("\nSTEP 4: Interpreting results...")
        
        interpreter = Interpreter(verbose=self.verbose)
        
        for result in results:
            if result['status'] == 'success':
                # Get LLM interpretation if available
                llm_interpretation = None
                if self.use_llm and self.reasoning_engine:
                    try:
                        llm_interpretation = self.reasoning_engine.interpret_results(
                            method_name=result['method'],
                            parameters=result['parameters'],
                            results=result['output'],
                            goal=goal
                        )
                    except Exception as e:
                        if self.verbose:
                            print(f"   Warning: LLM interpretation failed: {e}")
                
                interpretation = interpreter.interpret_result(
                    result,
                    method_name=result['method'],
                    llm_interpretation=llm_interpretation
                )
                
                # Print interpretation
                if self.verbose:
                    print(f"\n   {result['method']} Interpretation:")
                    for finding in interpretation.get('key_findings', []):
                        print(f"      - {finding}")
        
        # Step 5: Generate recommendations
        if self.verbose:
            print("\nSTEP 5: Generating recommendations...")
        
        recommendations = interpreter.suggest_next_steps(
            results, goal, data_profile
        )
        
        if self.verbose:
            print("\n   Recommended next steps:")
            for rec in recommendations:
                print(f"      - {rec}")
        
        # Create report
        reasoning_history = []
        if self.use_llm and self.reasoning_engine:
            reasoning_history = self.reasoning_engine.reasoning_history
        
        report = AnalysisReport(
            goal=goal,
            data_profile=data_profile,
            results=results,
            interpretations=interpreter.interpretations,
            reasoning_history=reasoning_history,
            recommendations=recommendations
        )
        
        # Store in history
        self.analysis_history.append(report)
        
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"\n{'='*70}")
            print(f"ANALYSIS COMPLETE ({elapsed:.2f} seconds)")
            print("="*70 + "\n")
        
        return report
    
    def _rule_based_method_selection(
        self,
        data_profile: Dict,
        goal: str,
        hypothesis: Optional[str] = None
    ) -> Dict:
        """
        Rule-based method selection (fallback when LLM unavailable).
        
        This uses statistical heuristics to select appropriate methods.
        """
        methods = []
        data_type = data_profile.get('data_type', 'unknown')
        n = data_profile.get('n', 0)
        
        # Distribution understanding
        if 'distribution' in goal.lower() or 'comprehensive' in goal.lower():
            if data_type == 'discrete_count':
                # Use Negative Binomial for count data
                mean = data_profile.get('mean', 10)
                var = data_profile.get('variance', 10)
                
                # Estimate p and k using method of moments
                if mean > 0 and var > mean:
                    # Overdispersed - good for Negative Binomial
                    p = mean / var
                    k = mean * p / (1 - p)
                    
                    methods.append({
                        "name": "NegativeBinomialAnalyzer",
                        "rationale": "Discrete count data with overdispersion detected",
                        "parameters": {"k": max(1, int(k)), "p": min(0.99, max(0.01, p))},
                        "expected_insights": "Distribution characteristics and probabilities"
                    })
        
        # Hypothesis testing
        if 'hypothesis' in goal.lower() or 'test' in goal.lower() or hypothesis:
            if data_type in ['continuous', 'continuous_positive', 'discrete']:
                # Z-test if sample size adequate
                if n >= 30:
                    mu_0 = 0
                    if hypothesis:
                        # Try to extract mu_0 from hypothesis
                        # Simple parsing for "mean > X" patterns
                        import re
                        match = re.search(r'(\d+\.?\d*)', hypothesis)
                        if match:
                            mu_0 = float(match.group(1))
                    
                    sigma = data_profile.get('std', 1)
                    
                    methods.append({
                        "name": "ZTest",
                        "rationale": "Testing hypothesis about population mean",
                        "parameters": {
                            "mu_0": mu_0,
                            "sigma": sigma,
                            "test_type": "two",
                            "alpha": 0.05
                        },
                        "expected_insights": "Statistical significance of hypothesis"
                    })
        
        # Parameter estimation
        if 'estimate' in goal.lower() or 'parameter' in goal.lower() or 'comprehensive' in goal.lower():
            if data_type in ['continuous', 'continuous_positive']:
                methods.append({
                    "name": "MethodOfMoments",
                    "rationale": "Estimating distribution parameters from data",
                    "parameters": {},
                    "expected_insights": "Parameter estimates for distribution"
                })
        
        # If no methods selected, provide a default
        if not methods:
            if data_type == 'discrete_count':
                methods.append({
                    "name": "NegativeBinomialAnalyzer",
                    "rationale": "Default analysis for count data",
                    "parameters": {"k": 10, "p": 0.5},
                    "expected_insights": "Basic distribution analysis"
                })
            else:
                methods.append({
                    "name": "MethodOfMoments",
                    "rationale": "Default parameter estimation",
                    "parameters": {},
                    "expected_insights": "Sample statistics and estimates"
                })
        
        return {
            "methods": methods,
            "analysis_workflow": "Rule-based method selection (no LLM)",
            "confidence": "medium"
        }
    
    def quick_analyze(self, data: Union[np.ndarray, list]) -> str:
        """
        Quick analysis with simplified output.
        
        Parameters
        ----------
        data : array_like
            Data to analyze
        
        Returns
        -------
        summary : str
            Brief analysis summary
        """
        report = self.analyze(data, goal="quick_overview")
        return report.summary()
    
    def explain_reasoning(self) -> str:
        """
        Explain the reasoning behind the last analysis.
        
        Returns
        -------
        explanation : str
            Detailed reasoning explanation
        """
        if not self.analysis_history:
            return "No analyses performed yet."
        
        if not self.use_llm or not self.reasoning_engine:
            return "Reasoning explanation only available when using LLM mode."
        
        return self.reasoning_engine.get_reasoning_summary()
    
    def get_last_report(self) -> Optional[AnalysisReport]:
        """Get the most recent analysis report."""
        if self.analysis_history:
            return self.analysis_history[-1]
        return None
    
    def reset(self):
        """Reset the agent's analysis history."""
        self.analysis_history = []
        if self.reasoning_engine:
            self.reasoning_engine.reasoning_history = []
        if self.verbose:
            print("Agent reset complete")

