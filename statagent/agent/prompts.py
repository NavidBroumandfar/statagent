"""
LLM prompt templates for the statistical reasoning engine.

This module contains all prompt templates used by the reasoning engine
to analyze data and make decisions about statistical methods.
"""

SYSTEM_PROMPT = """You are an expert statistician and data analyst with deep knowledge of:
- Probability distributions (discrete and continuous)
- Parameter estimation (Method of Moments, MLE, Bayesian)
- Hypothesis testing (Z-tests, t-tests, confidence intervals)
- Regression analysis (polynomial, regularized methods)
- Statistical inference and interpretation

Your role is to analyze data characteristics and recommend appropriate statistical methods.
Always explain your reasoning step-by-step. Be precise and scientific in your recommendations.
"""

DATA_ANALYSIS_PROMPT = """Analyze the following data profile and provide your expert assessment:

## Data Profile
- Sample size: {n}
- Data type: {data_type}
- Mean: {mean}
- Median: {median}
- Standard deviation: {std}
- Variance: {variance}
- Skewness: {skewness}
- Kurtosis: {kurtosis}
- Min: {min_val}
- Max: {max_val}
- Range: {range_val}
- Has zeros: {has_zeros}
- Has negatives: {has_negatives}
- Unique values: {unique_count}
- Missing values: {missing_count}

## Additional Characteristics
{additional_info}

## Analysis Goal
{goal}

Please provide:
1. **Data Type Assessment**: What type of data is this? (continuous, discrete, count data, etc.)
2. **Distribution Characteristics**: What does the data suggest about its underlying distribution?
3. **Recommended Methods**: Which statistical methods would be most appropriate?
4. **Reasoning**: Explain your recommendations step-by-step
5. **Warnings**: Any concerns or limitations to be aware of?

Format your response as a structured analysis.
"""

METHOD_SELECTION_PROMPT = """Based on the data analysis, select the most appropriate statistical methods:

## Data Summary
{data_summary}

## User Goal
{goal}

## Available Methods
1. **NegativeBinomialAnalyzer**: For discrete count data with overdispersion
   - Parameters: k (successes), p (probability)
   - Use when: Data is discrete counts, variance > mean
   
2. **SurvivalMixtureModel**: For continuous positive data (survival/waiting times)
   - Parameters: weights and rates for exponential mixture
   - Use when: Modeling time-to-event data
   
3. **MethodOfMoments**: Parameter estimation using sample moments
   - Parameters: Depends on distribution (e.g., k for Gamma)
   - Use when: Simple parameter estimation needed
   
4. **BayesianEstimator**: Bayesian parameter estimation with priors
   - Parameters: Prior parameters (alpha, beta)
   - Use when: Incorporating prior knowledge
   
5. **ZTest**: Hypothesis testing for means with known variance
   - Parameters: mu_0 (null hypothesis mean), sigma (known std)
   - Use when: Testing hypotheses about population mean
   
6. **PolynomialRegression**: Regression with polynomial features
   - Parameters: degree, lambda_ridge (regularization)
   - Use when: Modeling nonlinear relationships

## Your Task
Select 1-3 methods that would be most appropriate for this analysis.
For each method, provide:
1. Method name
2. Why it's appropriate
3. Suggested parameters (be specific)
4. Expected insights

Format as JSON:
{{
  "methods": [
    {{
      "name": "MethodName",
      "rationale": "Why this method is appropriate",
      "parameters": {{"param1": value1, "param2": value2}},
      "expected_insights": "What we'll learn"
    }}
  ],
  "analysis_workflow": "Step-by-step plan",
  "confidence": "high/medium/low"
}}
"""

PARAMETER_ESTIMATION_PROMPT = """Estimate parameters for the {method_name} method:

## Data Statistics
- Mean: {mean}
- Variance: {variance}
- Standard deviation: {std}
- Sample size: {n}

## Method Requirements
{method_requirements}

## Your Task
Based on the data statistics, estimate appropriate parameters for {method_name}.
Show your reasoning and calculations.

Provide response as JSON:
{{
  "parameters": {{"param1": value1, "param2": value2}},
  "reasoning": "Step-by-step explanation",
  "confidence": "high/medium/low",
  "warnings": ["Any warnings or concerns"]
}}
"""

RESULT_INTERPRETATION_PROMPT = """Interpret the following statistical analysis results:

## Analysis Performed
Method: {method_name}
Parameters: {parameters}

## Results
{results}

## Original Goal
{goal}

## Your Task
Provide a clear, insightful interpretation:
1. **Key Findings**: What do the results tell us?
2. **Statistical Significance**: Are the findings meaningful?
3. **Practical Implications**: What does this mean in practice?
4. **Limitations**: What are the caveats?
5. **Next Steps**: What analyses should follow?

Write in clear, accessible language suitable for both technical and non-technical audiences.
"""

HYPOTHESIS_GENERATION_PROMPT = """Based on the data characteristics, suggest testable hypotheses:

## Data Profile
{data_profile}

## Context
{context}

## Your Task
Generate 3-5 interesting hypotheses that could be tested with available methods.
For each hypothesis:
1. State the hypothesis clearly
2. Suggest the appropriate test
3. Explain why it's interesting
4. Estimate feasibility

Format as JSON array of hypotheses.
"""

ERROR_RECOVERY_PROMPT = """A statistical analysis encountered an error:

## Method Attempted
{method_name}

## Parameters Used
{parameters}

## Error
{error_message}

## Data Characteristics
{data_summary}

## Your Task
1. Diagnose why the method failed
2. Suggest alternative approaches
3. Recommend parameter adjustments
4. Provide a backup analysis plan

Format as JSON with alternative_methods array and reasoning.
"""

WORKFLOW_PLANNING_PROMPT = """Plan a multi-step statistical analysis workflow:

## Data Profile
{data_profile}

## Analysis Goal
{goal}

## Available Methods
- NegativeBinomialAnalyzer
- SurvivalMixtureModel
- MethodOfMoments
- BayesianEstimator
- ZTest
- PolynomialRegression

## Your Task
Create a step-by-step analysis workflow.
For each step, specify:
1. Method to use
2. Purpose of this step
3. Dependencies on previous steps
4. Expected output

Format as JSON:
{{
  "workflow": [
    {{
      "step": 1,
      "method": "MethodName",
      "purpose": "Why we're doing this",
      "parameters": {{}},
      "depends_on": [],
      "expected_output": "What we'll get"
    }}
  ],
  "overall_strategy": "High-level explanation",
  "estimated_time": "rough estimate"
}}
"""

CONFIDENCE_ASSESSMENT_PROMPT = """Assess confidence in the analysis results:

## Methods Used
{methods_used}

## Results Summary
{results_summary}

## Data Quality
- Sample size: {n}
- Missing data: {missing_pct}%
- Outliers detected: {outliers_count}

## Your Task
Provide a confidence assessment:
1. **Overall Confidence**: High/Medium/Low
2. **Reliability Factors**: What makes us confident?
3. **Concerns**: What limits our confidence?
4. **Recommendations**: How to improve confidence?

Be honest about limitations.
"""

def format_data_analysis_prompt(data_profile: dict, goal: str) -> str:
    """Format the data analysis prompt with actual data."""
    return DATA_ANALYSIS_PROMPT.format(
        n=data_profile.get('n', 'N/A'),
        data_type=data_profile.get('data_type', 'unknown'),
        mean=data_profile.get('mean', 'N/A'),
        median=data_profile.get('median', 'N/A'),
        std=data_profile.get('std', 'N/A'),
        variance=data_profile.get('variance', 'N/A'),
        skewness=data_profile.get('skewness', 'N/A'),
        kurtosis=data_profile.get('kurtosis', 'N/A'),
        min_val=data_profile.get('min', 'N/A'),
        max_val=data_profile.get('max', 'N/A'),
        range_val=data_profile.get('range', 'N/A'),
        has_zeros=data_profile.get('has_zeros', False),
        has_negatives=data_profile.get('has_negatives', False),
        unique_count=data_profile.get('unique_count', 'N/A'),
        missing_count=data_profile.get('missing_count', 0),
        additional_info=data_profile.get('additional_info', 'None'),
        goal=goal
    )

def format_method_selection_prompt(data_summary: str, goal: str) -> str:
    """Format the method selection prompt."""
    return METHOD_SELECTION_PROMPT.format(
        data_summary=data_summary,
        goal=goal
    )

def format_result_interpretation_prompt(method_name: str, parameters: dict, 
                                       results: str, goal: str) -> str:
    """Format the result interpretation prompt."""
    return RESULT_INTERPRETATION_PROMPT.format(
        method_name=method_name,
        parameters=parameters,
        results=results,
        goal=goal
    )

