"""
Orchestrator for executing statistical analysis workflows.

This module manages the execution of statistical methods, handles errors,
and coordinates between the reasoning engine and actual statagent tools.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable
import traceback
import warnings

# Import all statagent tools
from statagent.distributions import NegativeBinomialAnalyzer, SurvivalMixtureModel
from statagent.estimation import MethodOfMoments, BayesianEstimator
from statagent.inference import ZTest
from statagent.regression import PolynomialRegression


class Orchestrator:
    """
    Orchestrates execution of statistical analysis workflows.
    
    The Orchestrator manages tool execution, error handling, and result
    collection for autonomous statistical analysis.
    
    Parameters
    ----------
    data : np.ndarray
        Data to analyze
    verbose : bool, optional
        Whether to print execution progress (default: True)
    
    Attributes
    ----------
    data : np.ndarray
        Analysis data
    results : list
        Collected results from all executions
    execution_log : list
        Log of all execution attempts
    
    Examples
    --------
    >>> orchestrator = Orchestrator(data, verbose=True)
    >>> result = orchestrator.execute_method("ZTest", {"mu_0": 100, "sigma": 15})
    >>> results = orchestrator.execute_workflow(workflow_plan)
    """
    
    def __init__(self, data: np.ndarray, verbose: bool = True):
        """Initialize the orchestrator."""
        self.data = np.asarray(data)
        self.verbose = verbose
        self.results = []
        self.execution_log = []
        
        # Map method names to classes
        self.method_map = {
            "NegativeBinomialAnalyzer": NegativeBinomialAnalyzer,
            "SurvivalMixtureModel": SurvivalMixtureModel,
            "MethodOfMoments": MethodOfMoments,
            "BayesianEstimator": BayesianEstimator,
            "ZTest": ZTest,
            "PolynomialRegression": PolynomialRegression,
        }
    
    def execute_method(self, method_name: str, parameters: Dict,
                      operation: Optional[str] = None) -> Dict:
        """
        Execute a single statistical method.
        
        Parameters
        ----------
        method_name : str
            Name of the method to execute
        parameters : dict
            Parameters for the method
        operation : str, optional
            Specific operation to perform (e.g., "compute_statistics")
        
        Returns
        -------
        result : dict
            Execution result with status, output, and any errors
        """
        if self.verbose:
            print(f"Executing {method_name}...")
        
        result = {
            "method": method_name,
            "parameters": parameters,
            "status": "pending",
            "output": None,
            "error": None,
        }
        
        try:
            # Execute the method
            output = self._execute_statagent_method(
                method_name, parameters, operation
            )
            
            result["status"] = "success"
            result["output"] = output
            
            if self.verbose:
                print(f"{method_name} completed successfully")
        
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            
            if self.verbose:
                print(f"ERROR: {method_name} failed: {e}")
        
        # Log execution
        self.execution_log.append(result)
        self.results.append(result)
        
        return result
    
    def _execute_statagent_method(self, method_name: str, parameters: Dict,
                                 operation: Optional[str] = None) -> Any:
        """Execute a statagent method with given parameters."""
        
        # Get the method class
        method_class = self.method_map.get(method_name)
        
        if method_class is None:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Execute based on method type
        if method_name == "NegativeBinomialAnalyzer":
            return self._execute_negative_binomial(parameters)
        
        elif method_name == "SurvivalMixtureModel":
            return self._execute_survival_mixture(parameters)
        
        elif method_name == "MethodOfMoments":
            return self._execute_method_of_moments(parameters)
        
        elif method_name == "BayesianEstimator":
            return self._execute_bayesian(parameters)
        
        elif method_name == "ZTest":
            return self._execute_ztest(parameters)
        
        elif method_name == "PolynomialRegression":
            return self._execute_polynomial_regression(parameters)
        
        else:
            raise ValueError(f"Execution not implemented for {method_name}")
    
    def _execute_negative_binomial(self, params: Dict) -> Dict:
        """Execute Negative Binomial analysis."""
        k = params.get('k', 10)
        p = params.get('p', 0.5)
        
        analyzer = NegativeBinomialAnalyzer(k=k, p=p)
        
        results = {
            "statistics": analyzer.compute_statistics(),
            "significant_range": analyzer.find_significant_range(),
            "summary": analyzer.summary(),
        }
        
        return results
    
    def _execute_survival_mixture(self, params: Dict) -> Dict:
        """Execute Survival Mixture Model analysis."""
        w1 = params.get('w1', 0.5)
        rate1 = params.get('rate1', 1.0)
        w2 = params.get('w2', 0.5)
        rate2 = params.get('rate2', 2.0)
        
        model = SurvivalMixtureModel(w1=w1, rate1=rate1, w2=w2, rate2=rate2)
        
        results = {
            "statistics": model.compute_statistics(),
            "summary": model.summary(),
        }
        
        # If we have data, simulate and compare
        if len(self.data) > 0:
            samples = model.simulate(n=len(self.data))
            results["simulated_mean"] = float(np.mean(samples))
            results["data_mean"] = float(np.mean(self.data))
        
        return results
    
    def _execute_method_of_moments(self, params: Dict) -> Dict:
        """Execute Method of Moments estimation."""
        if len(self.data) == 0:
            raise ValueError("Method of Moments requires data")
        
        mom = MethodOfMoments(self.data)
        
        results = {
            "sample_mean": mom.sample_mean(),
            "sample_variance": mom.sample_variance(),
        }
        
        # If k parameter provided, estimate theta for Gamma
        if 'k' in params:
            k = params['k']
            theta_hat = mom.estimate_gamma_theta(k)
            results["gamma_k"] = k
            results["gamma_theta_hat"] = theta_hat
            results["expected_value"] = mom.expected_value_gamma(k, theta_hat)
        else:
            # Estimate both parameters
            both_params = mom.estimate_gamma_both_params()
            results["gamma_both_params"] = both_params
        
        results["summary"] = mom.summary(params.get('k'))
        
        return results
    
    def _execute_bayesian(self, params: Dict) -> Dict:
        """Execute Bayesian estimation."""
        alpha_prior = params.get('alpha_prior', 1)
        beta_prior = params.get('beta_prior', 1)
        
        estimator = BayesianEstimator(
            alpha_prior=alpha_prior,
            beta_prior=beta_prior
        )
        
        # If we have data, update with it
        if len(self.data) > 0:
            sample_mean = float(np.mean(self.data))
            n = len(self.data)
            k = params.get('k', 1)
            
            result = estimator.update_gamma_exponential(
                sample_mean=sample_mean,
                n=n,
                k=k
            )
            
            # Get credible interval
            interval = estimator.credible_interval(
                result['alpha_post'],
                result['beta_post']
            )
            
            result["credible_interval"] = interval
            result["summary"] = estimator.summary(sample_mean, n, k)
            
            return result
        else:
            return {
                "alpha_prior": alpha_prior,
                "beta_prior": beta_prior,
                "note": "No data provided for updating"
            }
    
    def _execute_ztest(self, params: Dict) -> Dict:
        """Execute Z-test."""
        if len(self.data) == 0:
            raise ValueError("Z-test requires data")
        
        mu_0 = params.get('mu_0', 0)
        sigma = params.get('sigma', None)
        
        # If sigma not provided, use sample std (though this isn't strictly a Z-test)
        if sigma is None:
            sigma = float(np.std(self.data, ddof=1))
            if sigma == 0:
                sigma = 1.0  # Avoid division by zero
        
        test = ZTest(self.data, mu_0=mu_0, sigma=sigma)
        
        # Determine test type
        test_type = params.get('test_type', 'two')
        alpha = params.get('alpha', 0.05)
        
        if test_type == 'left':
            result = test.left_tailed_test(alpha=alpha)
        elif test_type == 'right':
            result = test.right_tailed_test(alpha=alpha)
        else:
            result = test.two_tailed_test(alpha=alpha)
        
        # Add confidence interval
        ci = test.confidence_interval(1 - alpha)
        result["confidence_interval"] = ci
        result["summary"] = test.summary(test_type=test_type, alpha=alpha)
        
        return result
    
    def _execute_polynomial_regression(self, params: Dict) -> Dict:
        """Execute polynomial regression."""
        if len(self.data) == 0:
            raise ValueError("Polynomial regression requires data")
        
        # For regression, we need X and y
        # If data is 1D, create synthetic X or use indices
        if self.data.ndim == 1:
            X = np.arange(len(self.data))
            y = self.data
        elif self.data.ndim == 2:
            X = self.data[:, 0]
            y = self.data[:, 1]
        else:
            raise ValueError("Data must be 1D or 2D for regression")
        
        degree = params.get('degree', 3)
        lambda_ridge = params.get('lambda_ridge', 1e4)
        
        model = PolynomialRegression(degree=degree)
        model.fit(X, y, lambda_ridge=lambda_ridge)
        
        # Get predictions
        y_pred_ols = model.predict(X, method='ols')
        y_pred_ridge = model.predict(X, method='ridge')
        
        # Compute MSE
        mse_ols = model.compute_mse(X, y, method='ols')
        mse_ridge = model.compute_mse(X, y, method='ridge')
        
        results = {
            "degree": degree,
            "lambda_ridge": lambda_ridge,
            "mse_ols": mse_ols,
            "mse_ridge": mse_ridge,
            "coef_ols": model.coef_ols_.tolist() if model.coef_ols_ is not None else None,
            "coef_ridge": model.coef_ridge_.tolist() if model.coef_ridge_ is not None else None,
            "summary": model.summary(X, y),
        }
        
        return results
    
    def execute_workflow(self, workflow: Dict) -> List[Dict]:
        """
        Execute a planned workflow with multiple steps.
        
        Parameters
        ----------
        workflow : dict
            Workflow plan from ReasoningEngine
        
        Returns
        -------
        results : list
            Results from all workflow steps
        """
        if self.verbose:
            print("Executing workflow...")
            print(f"{'='*60}")
        
        workflow_results = []
        steps = workflow.get('workflow', [])
        
        for step in steps:
            step_num = step.get('step', '?')
            method = step.get('method', 'Unknown')
            parameters = step.get('parameters', {})
            
            if self.verbose:
                print(f"\nStep {step_num}: {method}")
                print(f"   Purpose: {step.get('purpose', 'N/A')}")
            
            # Execute the step
            result = self.execute_method(method, parameters)
            result['step_info'] = step
            workflow_results.append(result)
            
            # If step failed, consider stopping or retrying
            if result['status'] == 'failed' and self.verbose:
                print(f"   Warning: Step failed, continuing with workflow...")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Workflow execution complete")
        
        return workflow_results
    
    def execute_with_retry(self, method_name: str, parameters: Dict,
                          max_retries: int = 2) -> Dict:
        """
        Execute method with automatic retry on failure.
        
        Parameters
        ----------
        method_name : str
            Method to execute
        parameters : dict
            Method parameters
        max_retries : int, optional
            Maximum retry attempts (default: 2)
        
        Returns
        -------
        result : dict
            Final execution result
        """
        for attempt in range(max_retries + 1):
            if attempt > 0 and self.verbose:
                print(f"Retry attempt {attempt}/{max_retries}")
            
            result = self.execute_method(method_name, parameters)
            
            if result['status'] == 'success':
                return result
            
            # Try adjusting parameters for retry
            if attempt < max_retries:
                parameters = self._adjust_parameters_after_failure(
                    method_name, parameters, result['error']
                )
        
        return result
    
    def _adjust_parameters_after_failure(self, method_name: str,
                                        parameters: Dict, error: str) -> Dict:
        """Adjust parameters after a failure (simple heuristics)."""
        adjusted = parameters.copy()
        
        # Simple adjustments based on common errors
        if "division by zero" in error.lower():
            if 'sigma' in adjusted and adjusted['sigma'] == 0:
                adjusted['sigma'] = 1.0
        
        if "probability" in error.lower():
            if 'p' in adjusted:
                adjusted['p'] = max(0.01, min(0.99, adjusted['p']))
        
        return adjusted
    
    def get_successful_results(self) -> List[Dict]:
        """Get only successful results."""
        return [r for r in self.results if r['status'] == 'success']
    
    def get_failed_results(self) -> List[Dict]:
        """Get only failed results."""
        return [r for r in self.results if r['status'] == 'failed']
    
    def summary(self) -> str:
        """Generate execution summary."""
        total = len(self.execution_log)
        successful = len(self.get_successful_results())
        failed = len(self.get_failed_results())
        
        summary = f"""
Orchestrator Execution Summary
==============================
Total executions: {total}
Successful: {successful}
Failed: {failed}
Success rate: {successful/total*100:.1f}% (if total > 0)

Methods executed:
"""
        for result in self.execution_log:
            status_icon = "[OK]" if result['status'] == 'success' else "[FAIL]"
            summary += f"  {status_icon} {result['method']}\n"
        
        return summary

