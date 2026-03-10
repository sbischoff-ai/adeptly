# Repository Agent Instructions

This document applies to the full repository.

## Architecture map
- `adeptly/`: library source code.
  - `adeptly/agents/`: agent implementations (primary DQN agent).
  - `adeptly/observations/`: multimodal observation schemas and encoders.
  - `adeptly/trainer.py`: reusable training and real-time loop orchestration.
  - `adeptly/envs.py`, `adeptly/synthetic_env.py`: environment protocols and synthetic environments for local validation.
- `tests/`: unit and compatibility tests.
- `docs/`: generated documentation site artifacts.
- `.github/workflows/ci.yml`: CI matrix and required checks.

## Canonical command set (run with `uv run`)
Use these commands as the deterministic workflow for local and CI-safe checks:
- `uv run make format` — format Python code with Black.
- `uv run make lint` — check formatting and run static lint checks.
- `uv run make typecheck` — static typing with mypy.
- `uv run make test` — test suite (`pytest -q`).
- `uv run make docs` — rebuild documentation output.
- `uv run pre-commit run --all-files` — execute all pre-commit hooks.

## Required pre-finish validation
- Always run a type check before finishing.
- Standard pre-finish validation for code changes should include:
  - `uv run mypy adeptly tests`
  - `uv run pytest -q`

## Change constraints
- Keep changes minimal and scoped to the requested task.
- Do not weaken or skip tests to make checks pass.
- Do not modify build systems, dependencies, CI configuration, database schemas, or security-critical areas unless explicitly required.
- Update documentation (`README.md`, `CONTRIBUTING.md`, `AGENTS.md`) when workflows or contributor expectations change.
