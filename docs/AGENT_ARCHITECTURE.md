# StatAgent Phase 2: Agent Architecture

## Overview

StatAgent Phase 2 introduces an **autonomous agent layer** that transforms StatAgent from a traditional statistical library into an intelligent system capable of:

1. **Examining data** automatically
2. **Reasoning** about appropriate methods
3. **Executing** analyses autonomously
4. **Interpreting** results in plain language
5. **Recommending** next steps

## Architecture Components

### 1. StatisticalAgent (Main Interface)

**File**: `statagent/agent/statistical_agent.py`

The main user-facing class that orchestrates all components.

**Key Features**:
- Simple API: `agent.analyze(data, goal="...")`
- Supports both LLM-powered and rule-based analysis
- Maintains analysis history
- Provides detailed reports

**Usage**:
```python
from statagent import StatisticalAgent

agent = StatisticalAgent(llm="gpt-4", verbose=True)
report = agent.analyze(data, goal="understand_distribution")
```

**Parameters**:
- `llm`: LLM to use ("gpt-4", "gpt-3.5-turbo", "ollama/llama3")
- `api_key`: OpenAI API key (optional, reads from env)
- `verbose`: Print detailed progress
- `temperature`: LLM temperature for creativity
- `use_llm`: Enable/disable LLM (falls back to rules)

### 2. DataExaminer (Data Profiling)

**File**: `statagent/agent/data_examiner.py`

Performs comprehensive statistical profiling **without** requiring LLM calls.

**Capabilities**:
- Compute basic statistics (mean, median, std, etc.)
- Detect data type (discrete, continuous, count, etc.)
- Analyze distribution characteristics
- Check data quality (missing values, outliers)
- Detect patterns (multimodality, clustering)
- Generate preliminary recommendations

**Example Profile Output**:
```python
{
    'data_type': 'discrete_count',
    'n': 100,
    'mean': 23.45,
    'std': 8.32,
    'overdispersed': True,
    'has_outliers': True,
    'recommendations': [...]
}
```

**Key Methods**:
- `examine()`: Full data examination
- `summary()`: Human-readable summary
- Detection algorithms for type, outliers, patterns

### 3. ReasoningEngine (LLM-Powered Decisions)

**File**: `statagent/agent/reasoning_engine.py`

Uses LLMs to reason about data and make intelligent method selections.

**Supported LLMs**:
- **OpenAI**: GPT-4, GPT-3.5-Turbo (requires API key)
- **Ollama**: Local LLMs (llama3, mistral, etc.)

**Key Functions**:
- `analyze_data()`: Understand data characteristics
- `select_methods()`: Choose appropriate statistical methods
- `interpret_results()`: Translate results to insights
- `plan_workflow()`: Create multi-step analysis plans
- `recover_from_error()`: Suggest alternatives on failure

**Reasoning Process**:
1. Receives data profile from DataExaminer
2. Queries LLM with structured prompts
3. LLM reasons about data type and characteristics
4. Selects methods with parameters and rationale
5. Logs all reasoning steps for transparency

**Example LLM Interaction**:
```
User Goal: "understand_distribution"
Data Profile: {n: 100, mean: 23.45, variance: 69.2, ...}

LLM Reasoning:
- Data is discrete count type
- Variance (69.2) > Mean (23.45) indicates overdispersion
- Negative Binomial distribution appropriate
- Estimate parameters using method of moments
→ Selected: NegativeBinomialAnalyzer(k=7.9, p=0.252)
```

### 4. Orchestrator (Workflow Execution)

**File**: `statagent/agent/orchestrator.py`

Manages execution of statistical methods and handles errors.

**Responsibilities**:
- Execute statagent tools dynamically
- Handle method failures gracefully
- Collect results systematically
- Retry with adjusted parameters
- Log all execution attempts

**Method Mapping**:
```python
{
    "NegativeBinomialAnalyzer": NegativeBinomialAnalyzer,
    "SurvivalMixtureModel": SurvivalMixtureModel,
    "MethodOfMoments": MethodOfMoments,
    "BayesianEstimator": BayesianEstimator,
    "ZTest": ZTest,
    "PolynomialRegression": PolynomialRegression,
}
```

**Execution Flow**:
1. Receive method name and parameters
2. Instantiate appropriate class
3. Execute with data
4. Capture results or errors
5. Return structured result dict

**Error Handling**:
- Automatic retry with parameter adjustment
- Fallback to alternative methods
- Detailed error logging
- Graceful degradation

### 5. Interpreter (Result Translation)

**File**: `statagent/agent/interpreter.py`

Translates technical results into actionable insights.

**Capabilities**:
- Method-specific result interpretation
- Plain language explanations
- Statistical significance assessment
- Warning detection
- Next step recommendations

**Interpretation Structure**:
```python
{
    "method": "ZTest",
    "key_findings": [
        "Sample mean: 1050.23",
        "p-value: 0.0023"
    ],
    "insights": [
        "Strong statistical significance detected",
        "Evidence supports hypothesis"
    ],
    "warnings": ["Small sample size may limit generalizability"]
}
```

**Intelligence Features**:
- Detects practical vs statistical significance
- Compares estimates across methods
- Identifies potential issues
- Suggests validation approaches

### 6. Prompts Module

**File**: `statagent/agent/prompts.py`

Contains all LLM prompt templates for reasoning.

**Key Prompts**:
- `SYSTEM_PROMPT`: Agent identity and expertise
- `DATA_ANALYSIS_PROMPT`: Data understanding
- `METHOD_SELECTION_PROMPT`: Method selection with parameters
- `RESULT_INTERPRETATION_PROMPT`: Result explanation
- `WORKFLOW_PLANNING_PROMPT`: Multi-step planning
- `ERROR_RECOVERY_PROMPT`: Failure handling

**Design Principles**:
- Structured output (JSON when needed)
- Clear examples of available methods
- Request for step-by-step reasoning
- Emphasis on statistical rigor

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│  User: agent.analyze(data, goal="...")                 │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Data Examination (DataExaminer)               │
│  - Compute statistics                                   │
│  - Detect data type                                     │
│  - Analyze distribution                                 │
│  - Check quality                                        │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2: Method Selection (ReasoningEngine)            │
│  - Analyze data profile                                 │
│  - Reason about appropriate methods                     │
│  - Select methods with parameters                       │
│  - Provide rationale                                    │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3: Execution (Orchestrator)                      │
│  - Execute each selected method                         │
│  - Handle errors and retries                            │
│  - Collect results                                      │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4: Interpretation (Interpreter + ReasoningEngine)│
│  - Translate results to insights                        │
│  - LLM interpretation (if enabled)                      │
│  - Generate recommendations                             │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5: Report Generation (AnalysisReport)            │
│  - Compile all results                                  │
│  - Include reasoning history                            │
│  - Provide recommendations                              │
│  - Return to user                                       │
└─────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Autonomy
- Agent makes decisions with minimal user input
- Intelligent method selection based on data
- Automatic parameter estimation
- Self-correcting on errors

### 2. Transparency
- All reasoning steps logged
- Rationale provided for decisions
- Explainable method selection
- Accessible reasoning history

### 3. Robustness
- Graceful fallback to rule-based mode
- Error recovery mechanisms
- Parameter validation
- Multiple retry strategies

### 4. Flexibility
- Supports multiple LLMs (OpenAI, Ollama)
- Can run without LLM (rule-based)
- Extensible prompt system
- Configurable verbosity

### 5. Backward Compatibility
- Existing statagent tools unchanged
- Agent layer is optional
- Traditional API still available
- No breaking changes

## LLM Integration

### OpenAI Setup

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-key-here'

agent = StatisticalAgent(llm="gpt-4")
```

### Ollama Setup (Local)

1. Install Ollama: https://ollama.ai
2. Pull model: `ollama pull llama3`
3. Use in agent:

```python
agent = StatisticalAgent(llm="ollama/llama3")
```

### Rule-Based Mode (No LLM)

```python
agent = StatisticalAgent(use_llm=False)
# Uses statistical heuristics instead of LLM
```

## Extension Points

### Adding New Statistical Methods

1. Add method to `statagent/` (distributions, estimation, etc.)
2. Register in Orchestrator's `method_map`
3. Add execution logic in Orchestrator
4. Add interpretation logic in Interpreter
5. Update prompts to include new method

### Adding New LLM Providers

1. Add setup method in ReasoningEngine
2. Implement call method
3. Handle response format
4. Add to documentation

### Custom Prompts

```python
from statagent.agent.prompts import DATA_ANALYSIS_PROMPT

# Modify prompt templates
custom_prompt = DATA_ANALYSIS_PROMPT + "\nAdditional instruction..."
```

## Performance Considerations

### LLM Calls
- Data profiling: 0 LLM calls
- Method selection: 1-2 LLM calls
- Result interpretation: 1 call per method
- Total: ~3-5 LLM calls per analysis

### Optimization Strategies
1. Cache data profiles
2. Batch LLM requests
3. Use rule-based mode for simple cases
4. Adjust temperature for speed vs quality

### Costs (Approximate)
- GPT-4: $0.03 - $0.10 per analysis
- GPT-3.5-Turbo: $0.001 - $0.01 per analysis
- Ollama: Free (local compute)

## Security & Privacy

### API Key Management
- Read from environment variables
- Never log or display keys
- Support for multiple providers

### Data Privacy
- Data sent to LLM only as statistics (not raw data)
- Option to use local LLM (Ollama)
- Rule-based mode requires no external calls

## Testing

### Unit Tests
```bash
pytest tests/agent/test_data_examiner.py
pytest tests/agent/test_orchestrator.py
```

### Integration Tests
```bash
pytest tests/agent/test_statistical_agent.py
```

### Mock LLM Tests
```python
# Use mock responses for CI/CD
agent = StatisticalAgent(use_llm=False)
```

## Future Enhancements

### Phase 3 Possibilities
1. **Memory System**: Remember past analyses
2. **Interactive Mode**: Ask user for clarification
3. **Multi-Agent**: Specialized agents for domains
4. **Visualization Agent**: Auto-generate plots
5. **Report Generation**: PDF/HTML reports
6. **Streaming**: Real-time analysis updates
7. **Confidence Scores**: Express certainty about decisions
8. **A/B Testing**: Compare multiple methods
9. **Causal Inference**: Advanced causal analysis
10. **Active Learning**: Suggest data collection

## Troubleshooting

### LLM Not Available
- Check API key: `echo $OPENAI_API_KEY`
- Verify package: `pip show openai`
- Use rule-based mode: `use_llm=False`

### Method Execution Fails
- Check data format (1D array expected)
- Verify parameter ranges
- Review error logs in report
- Try alternative methods

### Poor Method Selection
- Provide clearer goal description
- Use more specific hypothesis
- Check data profile accuracy
- Adjust LLM temperature

## References

- StatAgent Core: `docs/API_REFERENCE.md`
- Examples: `examples/agent_autonomous_analysis.py`
- Prompt Engineering: `statagent/agent/prompts.py`
- Contributing: `docs/CONTRIBUTING.md`

---

**Built with autonomy, transparency, and statistical rigor.**

