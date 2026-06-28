# StatAgent Working Notes

## Positioning

StatAgent is an experimental statistical-analysis toolkit and portfolio project.
Keep public claims modest: the agent layer is a prototype for agent-assisted
method selection, not a production-grade autonomous statistician.

## Development Commands

Use a local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verification:

```bash
pytest
black --check statagent tests setup.py examples/portfolio_demo.py examples/agent_examples.py
flake8 statagent/ tests/ examples/portfolio_demo.py examples/agent_examples.py setup.py
python examples/portfolio_demo.py
```

## Project Hygiene

- Keep LLM dependencies optional through `.[llm]`.
- Keep notebook dependencies optional through `.[notebooks]`.
- Do not add private scratch files, local paths, or employer/client references.
- Preserve the legacy organization-name sanitization boundary from the audit:
  no traces should appear in tracked files.
