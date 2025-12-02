"""
Reasoning Engine for LLM-powered statistical decision making.

This module uses LLMs (OpenAI or Ollama) to reason about data characteristics
and make intelligent decisions about statistical methods.
"""

import json
import os
from typing import Dict, List, Optional, Union, Any
import warnings

from statagent.agent.prompts import (
    SYSTEM_PROMPT,
    format_data_analysis_prompt,
    format_method_selection_prompt,
    format_result_interpretation_prompt,
    WORKFLOW_PLANNING_PROMPT,
    PARAMETER_ESTIMATION_PROMPT,
    ERROR_RECOVERY_PROMPT,
)


class ReasoningEngine:
    """
    LLM-powered reasoning engine for statistical analysis.
    
    The ReasoningEngine uses large language models to analyze data profiles,
    select appropriate methods, and interpret results.
    
    Parameters
    ----------
    llm : str, optional
        LLM to use: "gpt-4", "gpt-3.5-turbo", "ollama/llama3", etc.
        Default: "gpt-4"
    api_key : str, optional
        OpenAI API key (if None, reads from OPENAI_API_KEY env var)
    verbose : bool, optional
        Whether to print reasoning steps (default: True)
    temperature : float, optional
        LLM temperature for creativity (default: 0.1 for consistency)
    
    Attributes
    ----------
    llm : str
        LLM identifier
    verbose : bool
        Verbosity flag
    reasoning_history : list
        History of all reasoning steps
    
    Examples
    --------
    >>> engine = ReasoningEngine(llm="gpt-4", verbose=True)
    >>> analysis = engine.analyze_data(data_profile, goal="understand_distribution")
    >>> methods = engine.select_methods(data_profile, goal)
    """
    
    def __init__(
        self,
        llm: str = "gpt-4",
        api_key: Optional[str] = None,
        verbose: bool = True,
        temperature: float = 0.1,
    ):
        """Initialize the reasoning engine."""
        self.llm = llm
        self.verbose = verbose
        self.temperature = temperature
        self.reasoning_history = []
        
        # Setup LLM client
        if llm.startswith("gpt"):
            self._setup_openai(api_key)
        elif llm.startswith("ollama"):
            self._setup_ollama()
        else:
            raise ValueError(f"Unsupported LLM: {llm}")
    
    def _setup_openai(self, api_key: Optional[str] = None):
        """Setup OpenAI client."""
        try:
            import openai
            self.client_type = "openai"
            
            # Get API key
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
            
            if api_key is None:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY environment "
                    "variable or pass api_key parameter."
                )
            
            # Initialize client (works for both openai v0.x and v1.x)
            try:
                # Try v1.x style
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                # Fall back to v0.x style
                openai.api_key = api_key
                self.client = openai
                
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
    
    def _setup_ollama(self):
        """Setup Ollama client."""
        try:
            import ollama
            self.client_type = "ollama"
            self.client = ollama
            # Extract model name (e.g., "ollama/llama3" -> "llama3")
            self.ollama_model = self.llm.split("/", 1)[1] if "/" in self.llm else "llama3"
        except ImportError:
            raise ImportError(
                "Ollama package not installed. Install with: pip install ollama"
            )
    
    def _call_llm(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        """Call the LLM with messages."""
        try:
            if self.client_type == "openai":
                return self._call_openai(messages, json_mode)
            elif self.client_type == "ollama":
                return self._call_ollama(messages, json_mode)
        except Exception as e:
            if self.verbose:
                print(f"Warning: LLM call failed: {e}")
            return self._fallback_response()
    
    def _call_openai(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        """Call OpenAI API."""
        try:
            # Try v1.x style first
            kwargs = {
                "model": self.llm,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
            
        except AttributeError:
            # Fall back to v0.x style
            kwargs = {
                "model": self.llm,
                "messages": messages,
                "temperature": self.temperature,
            }
            response = self.client.ChatCompletion.create(**kwargs)
            return response.choices[0].message.content
    
    def _call_ollama(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        """Call Ollama API."""
        response = self.client.chat(
            model=self.ollama_model,
            messages=messages,
            options={"temperature": self.temperature},
        )
        return response['message']['content']
    
    def _fallback_response(self) -> str:
        """Provide fallback response when LLM fails."""
        return json.dumps({
            "error": "LLM call failed",
            "fallback": True,
            "message": "Using rule-based fallback"
        })
    
    def analyze_data(self, data_profile: Dict, goal: str) -> Dict:
        """
        Analyze data characteristics and provide assessment.
        
        Parameters
        ----------
        data_profile : dict
            Data profile from DataExaminer
        goal : str
            Analysis goal (e.g., "understand_distribution")
        
        Returns
        -------
        analysis : dict
            LLM analysis of data characteristics
        """
        if self.verbose:
            print("Analyzing data characteristics...")
        
        # Format prompt
        prompt = format_data_analysis_prompt(data_profile, goal)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Call LLM
        response = self._call_llm(messages)
        
        # Log reasoning
        self.reasoning_history.append({
            "step": "data_analysis",
            "input": {"profile": data_profile, "goal": goal},
            "output": response
        })
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("DATA ANALYSIS")
            print('='*60)
            print(response)
            print('='*60 + '\n')
        
        return {"analysis": response, "raw_response": response}
    
    def select_methods(self, data_profile: Dict, goal: str, 
                      data_analysis: Optional[str] = None) -> Dict:
        """
        Select appropriate statistical methods based on data.
        
        Parameters
        ----------
        data_profile : dict
            Data profile from DataExaminer
        goal : str
            Analysis goal
        data_analysis : str, optional
            Previous data analysis (if available)
        
        Returns
        -------
        methods : dict
            Selected methods with parameters and rationale
        """
        if self.verbose:
            print("Selecting statistical methods...")
        
        # Create data summary
        if data_analysis:
            data_summary = data_analysis
        else:
            basic = data_profile.get('basic_stats', {})
            data_summary = f"""
            Sample size: {basic.get('n', 0)}
            Data type: {data_profile.get('data_type', 'unknown')}
            Mean: {basic.get('mean', 0):.4f}
            Std: {basic.get('std', 0):.4f}
            """
        
        # Format prompt
        prompt = format_method_selection_prompt(data_summary, goal)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Call LLM with JSON mode
        response = self._call_llm(messages, json_mode=True)
        
        # Parse JSON response
        try:
            methods = json.loads(response)
        except json.JSONDecodeError:
            if self.verbose:
                print("Warning: Failed to parse JSON, using fallback")
            methods = self._fallback_method_selection(data_profile)
        
        # Log reasoning
        self.reasoning_history.append({
            "step": "method_selection",
            "input": {"profile": data_profile, "goal": goal},
            "output": methods
        })
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("METHOD SELECTION")
            print('='*60)
            print(json.dumps(methods, indent=2))
            print('='*60 + '\n')
        
        return methods
    
    def interpret_results(self, method_name: str, parameters: Dict,
                         results: Any, goal: str) -> str:
        """
        Interpret analysis results in plain language.
        
        Parameters
        ----------
        method_name : str
            Name of method used
        parameters : dict
            Parameters used
        results : any
            Analysis results
        goal : str
            Original analysis goal
        
        Returns
        -------
        interpretation : str
            Plain language interpretation
        """
        if self.verbose:
            print("Interpreting results...")
        
        # Format results as string
        if isinstance(results, dict):
            results_str = json.dumps(results, indent=2)
        else:
            results_str = str(results)
        
        # Format prompt
        prompt = format_result_interpretation_prompt(
            method_name, parameters, results_str, goal
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Call LLM
        interpretation = self._call_llm(messages)
        
        # Log reasoning
        self.reasoning_history.append({
            "step": "interpretation",
            "input": {
                "method": method_name,
                "parameters": parameters,
                "results": results_str
            },
            "output": interpretation
        })
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("INTERPRETATION")
            print('='*60)
            print(interpretation)
            print('='*60 + '\n')
        
        return interpretation
    
    def plan_workflow(self, data_profile: Dict, goal: str) -> Dict:
        """
        Plan a multi-step analysis workflow.
        
        Parameters
        ----------
        data_profile : dict
            Data profile
        goal : str
            Analysis goal
        
        Returns
        -------
        workflow : dict
            Planned workflow with steps
        """
        if self.verbose:
            print("Planning analysis workflow...")
        
        prompt = WORKFLOW_PLANNING_PROMPT.format(
            data_profile=json.dumps(data_profile, indent=2),
            goal=goal
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_llm(messages, json_mode=True)
        
        try:
            workflow = json.loads(response)
        except json.JSONDecodeError:
            workflow = {"error": "Failed to parse workflow"}
        
        self.reasoning_history.append({
            "step": "workflow_planning",
            "input": {"profile": data_profile, "goal": goal},
            "output": workflow
        })
        
        return workflow
    
    def recover_from_error(self, method_name: str, parameters: Dict,
                          error: str, data_profile: Dict) -> Dict:
        """
        Suggest alternatives when a method fails.
        
        Parameters
        ----------
        method_name : str
            Method that failed
        parameters : dict
            Parameters used
        error : str
            Error message
        data_profile : dict
            Data profile
        
        Returns
        -------
        recovery : dict
            Recovery suggestions
        """
        if self.verbose:
            print(f"Recovering from error in {method_name}...")
        
        prompt = ERROR_RECOVERY_PROMPT.format(
            method_name=method_name,
            parameters=json.dumps(parameters),
            error_message=error,
            data_summary=json.dumps(data_profile.get('basic_stats', {}))
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_llm(messages, json_mode=True)
        
        try:
            recovery = json.loads(response)
        except json.JSONDecodeError:
            recovery = {"error": "Failed to parse recovery plan"}
        
        return recovery
    
    def _fallback_method_selection(self, data_profile: Dict) -> Dict:
        """Rule-based fallback when LLM fails."""
        data_type = data_profile.get('data_type', 'unknown')
        
        methods = []
        
        if data_type == 'discrete_count':
            methods.append({
                "name": "NegativeBinomialAnalyzer",
                "rationale": "Discrete count data detected (fallback selection)",
                "parameters": {"k": 10, "p": 0.5},
                "expected_insights": "Distribution characteristics"
            })
        elif data_type == 'continuous':
            methods.append({
                "name": "ZTest",
                "rationale": "Continuous data - hypothesis testing (fallback)",
                "parameters": {"mu_0": 0, "sigma": 1},
                "expected_insights": "Statistical significance"
            })
        
        return {
            "methods": methods,
            "analysis_workflow": "Fallback workflow due to LLM failure",
            "confidence": "low"
        }
    
    def get_reasoning_summary(self) -> str:
        """Get a summary of all reasoning steps."""
        summary = "Reasoning History\n" + "="*60 + "\n\n"
        
        for i, step in enumerate(self.reasoning_history, 1):
            summary += f"Step {i}: {step['step']}\n"
            summary += f"{'─'*60}\n"
            
            if 'output' in step:
                if isinstance(step['output'], dict):
                    summary += json.dumps(step['output'], indent=2) + "\n"
                else:
                    summary += str(step['output']) + "\n"
            
            summary += "\n"
        
        return summary

