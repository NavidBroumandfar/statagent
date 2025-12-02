"""
Interpreter for translating statistical results into insights.

This module interprets analysis results and generates actionable
recommendations in plain language.
"""

import json
from typing import Dict, List, Any, Optional
import numpy as np


class Interpreter:
    """
    Interprets statistical analysis results.
    
    The Interpreter translates technical statistical results into
    plain language insights and recommendations.
    
    Parameters
    ----------
    verbose : bool, optional
        Whether to print interpretation progress (default: True)
    
    Attributes
    ----------
    interpretations : list
        History of all interpretations
    
    Examples
    --------
    >>> interpreter = Interpreter(verbose=True)
    >>> insights = interpreter.interpret_result(result, method_name="ZTest")
    >>> recommendations = interpreter.suggest_next_steps(all_results, goal)
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize the interpreter."""
        self.verbose = verbose
        self.interpretations = []
    
    def interpret_result(self, result: Dict, method_name: str,
                        llm_interpretation: Optional[str] = None) -> Dict:
        """
        Interpret a single analysis result.
        
        Parameters
        ----------
        result : dict
            Result from Orchestrator execution
        method_name : str
            Name of method that produced result
        llm_interpretation : str, optional
            LLM-generated interpretation (if available)
        
        Returns
        -------
        interpretation : dict
            Structured interpretation with key findings and insights
        """
        if self.verbose:
            print(f"Interpreting {method_name} results...")
        
        interpretation = {
            "method": method_name,
            "status": result.get('status', 'unknown'),
            "key_findings": [],
            "insights": [],
            "warnings": [],
            "llm_interpretation": llm_interpretation,
        }
        
        if result['status'] != 'success':
            interpretation['warnings'].append(
                f"Analysis failed: {result.get('error', 'Unknown error')}"
            )
            return interpretation
        
        # Method-specific interpretation
        output = result.get('output', {})
        
        if method_name == "NegativeBinomialAnalyzer":
            interpretation.update(self._interpret_negative_binomial(output))
        elif method_name == "SurvivalMixtureModel":
            interpretation.update(self._interpret_survival_mixture(output))
        elif method_name == "MethodOfMoments":
            interpretation.update(self._interpret_method_of_moments(output))
        elif method_name == "BayesianEstimator":
            interpretation.update(self._interpret_bayesian(output))
        elif method_name == "ZTest":
            interpretation.update(self._interpret_ztest(output))
        elif method_name == "PolynomialRegression":
            interpretation.update(self._interpret_polynomial_regression(output))
        
        self.interpretations.append(interpretation)
        return interpretation
    
    def _interpret_negative_binomial(self, output: Dict) -> Dict:
        """Interpret Negative Binomial results."""
        stats = output.get('statistics', {})
        
        findings = [
            f"Mean number of events: {stats.get('mean', 'N/A'):.2f}",
            f"Median: {stats.get('median', 'N/A')}",
            f"Standard deviation: {stats.get('std', 'N/A'):.2f}",
        ]
        
        insights = [
            "This represents a discrete count process.",
        ]
        
        # Check for overdispersion
        mean = stats.get('mean', 0)
        var = stats.get('variance', 0)
        if var > mean * 1.5:
            insights.append(
                f"Data shows overdispersion (variance {var:.2f} > mean {mean:.2f}), "
                "which the Negative Binomial distribution handles well."
            )
        
        return {"key_findings": findings, "insights": insights}
    
    def _interpret_survival_mixture(self, output: Dict) -> Dict:
        """Interpret Survival Mixture Model results."""
        stats = output.get('statistics', {})
        
        findings = [
            f"Expected waiting time: {stats.get('mean', 'N/A'):.4f}",
        ]
        
        insights = [
            "The mixture model captures heterogeneity in the process,",
            "suggesting multiple underlying populations or states.",
        ]
        
        return {"key_findings": findings, "insights": insights}
    
    def _interpret_method_of_moments(self, output: Dict) -> Dict:
        """Interpret Method of Moments results."""
        findings = [
            f"Sample mean: {output.get('sample_mean', 'N/A'):.4f}",
            f"Sample variance: {output.get('sample_variance', 'N/A'):.4f}",
        ]
        
        insights = []
        
        if 'gamma_theta_hat' in output:
            findings.append(
                f"Estimated scale parameter (theta): {output['gamma_theta_hat']:.4f}"
            )
            insights.append(
                "Method of Moments provides a simple, classical estimate "
                "that matches sample moments to theoretical moments."
            )
        
        if 'gamma_both_params' in output:
            params = output['gamma_both_params']
            findings.append(
                f"Estimated shape (k): {params['k']:.4f}, "
                f"scale (theta): {params['theta']:.4f}"
            )
        
        return {"key_findings": findings, "insights": insights}
    
    def _interpret_bayesian(self, output: Dict) -> Dict:
        """Interpret Bayesian estimation results."""
        findings = []
        insights = []
        
        if 'bayes_estimate' in output:
            findings.append(
                f"Posterior mean (Bayes estimate): {output['bayes_estimate']:.4f}"
            )
        
        if 'map_estimate' in output:
            findings.append(
                f"MAP estimate: {output['map_estimate']:.4f}"
            )
        
        if 'mle_estimate' in output:
            findings.append(
                f"MLE estimate: {output['mle_estimate']:.4f}"
            )
        
        if 'credible_interval' in output:
            ci = output['credible_interval']
            findings.append(
                f"95% credible interval: [{ci[0]:.4f}, {ci[1]:.4f}]"
            )
            insights.append(
                "The credible interval quantifies uncertainty about the parameter, "
                "incorporating both prior knowledge and observed data."
            )
        
        # Check if prior was influential
        if all(k in output for k in ['alpha_prior', 'alpha_post', 'beta_prior', 'beta_post']):
            prior_mean = output['alpha_prior'] / output['beta_prior']
            post_mean = output.get('bayes_estimate', 0)
            
            if abs(prior_mean - post_mean) / prior_mean > 0.2:
                insights.append(
                    "The data substantially updated the prior belief."
                )
            else:
                insights.append(
                    "The posterior is relatively close to the prior, "
                    "suggesting either strong prior or limited data."
                )
        
        return {"key_findings": findings, "insights": insights}
    
    def _interpret_ztest(self, output: Dict) -> Dict:
        """Interpret Z-test results."""
        findings = [
            f"Sample mean: {output.get('sample_mean', 'N/A'):.4f}",
            f"Z-statistic: {output.get('z_statistic', 'N/A'):.4f}",
            f"p-value: {output.get('p_value', 'N/A'):.6f}",
        ]
        
        insights = []
        warnings = []
        
        reject = output.get('reject_null', False)
        alpha = output.get('alpha', 0.05)
        p_value = output.get('p_value', 1.0)
        
        if reject:
            insights.append(
                f"Statistical significance detected (p < {alpha}). "
                "The null hypothesis is rejected."
            )
            
            if p_value < 0.001:
                insights.append("The evidence is very strong.")
            elif p_value < 0.01:
                insights.append("The evidence is strong.")
            else:
                insights.append("The evidence is moderate.")
        else:
            insights.append(
                f"No statistical significance detected (p ≥ {alpha}). "
                "Insufficient evidence to reject the null hypothesis."
            )
            
            if p_value > 0.5:
                insights.append(
                    "The p-value is quite large, suggesting the data are "
                    "consistent with the null hypothesis."
                )
        
        # Confidence interval interpretation
        if 'confidence_interval' in output:
            ci = output['confidence_interval']
            findings.append(
                f"{int((1-alpha)*100)}% confidence interval: "
                f"[{ci[0]:.2f}, {ci[1]:.2f}]"
            )
        
        return {
            "key_findings": findings,
            "insights": insights,
            "warnings": warnings
        }
    
    def _interpret_polynomial_regression(self, output: Dict) -> Dict:
        """Interpret Polynomial Regression results."""
        findings = [
            f"Polynomial degree: {output.get('degree', 'N/A')}",
            f"OLS MSE: {output.get('mse_ols', 'N/A'):.2e}",
            f"Ridge MSE: {output.get('mse_ridge', 'N/A'):.2e}",
        ]
        
        insights = []
        warnings = []
        
        mse_ols = output.get('mse_ols', float('inf'))
        mse_ridge = output.get('mse_ridge', float('inf'))
        
        if np.isfinite(mse_ols) and np.isfinite(mse_ridge):
            if mse_ridge < mse_ols * 0.95:
                insights.append(
                    "Ridge regression significantly outperforms OLS, "
                    "suggesting the regularization helps prevent overfitting."
                )
            elif mse_ols < mse_ridge * 0.95:
                insights.append(
                    "OLS performs better than Ridge, suggesting the "
                    "regularization may be too strong or unnecessary."
                )
            else:
                insights.append(
                    "OLS and Ridge perform similarly on this data."
                )
        
        if np.isinf(mse_ols):
            warnings.append(
                "OLS failed (likely numerical instability). "
                "Use Ridge regression results."
            )
        
        degree = output.get('degree', 0)
        if degree > 10:
            warnings.append(
                f"High polynomial degree ({degree}) may lead to overfitting. "
                "Consider regularization or lower degree."
            )
        
        return {
            "key_findings": findings,
            "insights": insights,
            "warnings": warnings
        }
    
    def suggest_next_steps(self, results: List[Dict], goal: str,
                          data_profile: Optional[Dict] = None) -> List[str]:
        """
        Suggest next analysis steps based on results.
        
        Parameters
        ----------
        results : list
            All analysis results so far
        goal : str
            Original analysis goal
        data_profile : dict, optional
            Data profile from DataExaminer
        
        Returns
        -------
        suggestions : list
            List of suggested next steps
        """
        suggestions = []
        
        # Check what methods have been run
        methods_run = set(r['method'] for r in results if r['status'] == 'success')
        
        # Suggest based on what's been done
        if "ZTest" in methods_run:
            if "BayesianEstimator" not in methods_run:
                suggestions.append(
                    "Consider Bayesian estimation to incorporate prior knowledge"
                )
        
        if "MethodOfMoments" in methods_run:
            if "BayesianEstimator" not in methods_run:
                suggestions.append(
                    "Compare with Bayesian estimation for uncertainty quantification"
                )
        
        if "NegativeBinomialAnalyzer" in methods_run:
            suggestions.append(
                "Consider goodness-of-fit tests to validate the distribution"
            )
        
        # Data-driven suggestions
        if data_profile:
            data_type = data_profile.get('data_type', '')
            
            if data_type == 'continuous' and not any('Regression' in m for m in methods_run):
                suggestions.append(
                    "Explore relationships with regression analysis"
                )
        
        # Generic suggestions
        if not suggestions:
            suggestions.append("Explore alternative distributions or models")
            suggestions.append("Perform sensitivity analysis on parameters")
            suggestions.append("Validate results with simulation studies")
        
        return suggestions
    
    def generate_report(self, all_results: List[Dict],
                       goal: str, data_profile: Dict) -> str:
        """
        Generate comprehensive analysis report.
        
        Parameters
        ----------
        all_results : list
            All analysis results
        goal : str
            Analysis goal
        data_profile : dict
            Data profile
        
        Returns
        -------
        report : str
            Formatted analysis report
        """
        report = f"""
{'='*70}
STATISTICAL ANALYSIS REPORT
{'='*70}

ANALYSIS GOAL
-------------
{goal}

DATA SUMMARY
------------
Sample size: {data_profile.get('n', 'N/A')}
Data type: {data_profile.get('data_type', 'unknown')}
Mean: {data_profile.get('mean', 'N/A'):.4f}
Std Dev: {data_profile.get('std', 'N/A'):.4f}
Range: [{data_profile.get('min', 'N/A'):.2f}, {data_profile.get('max', 'N/A'):.2f}]

ANALYSES PERFORMED
------------------
"""
        
        # Add each analysis
        for i, result in enumerate(all_results, 1):
            method = result.get('method', 'Unknown')
            status = result.get('status', 'unknown')
            
            report += f"\n{i}. {method}\n"
            report += f"   Status: {status}\n"
            
            if status == 'success':
                # Get interpretation
                interpretation = next(
                    (interp for interp in self.interpretations 
                     if interp['method'] == method),
                    None
                )
                
                if interpretation:
                    report += "   Key Findings:\n"
                    for finding in interpretation.get('key_findings', []):
                        report += f"     • {finding}\n"
                    
                    if interpretation.get('insights'):
                        report += "   Insights:\n"
                        for insight in interpretation['insights']:
                            report += f"     • {insight}\n"
            else:
                report += f"   Error: {result.get('error', 'Unknown')}\n"
        
        # Add recommendations
        suggestions = self.suggest_next_steps(all_results, goal, data_profile)
        
        report += f"""
RECOMMENDATIONS
---------------
"""
        for suggestion in suggestions:
            report += f"• {suggestion}\n"
        
        report += f"\n{'='*70}\n"
        
        return report
    
    def get_key_insights(self) -> List[str]:
        """Extract all key insights from interpretations."""
        insights = []
        
        for interp in self.interpretations:
            insights.extend(interp.get('insights', []))
        
        return insights
    
    def get_all_warnings(self) -> List[str]:
        """Extract all warnings from interpretations."""
        warnings = []
        
        for interp in self.interpretations:
            warnings.extend(interp.get('warnings', []))
        
        return warnings

