# Contributing to Adeptly

Thanks for contributing to Adeptly.

## Development environment

```bash
uv python install 3.12
uv venv --python 3.12
uv sync --frozen --extra dev
```

## Branch and commit conventions

### Branch naming
Use short, descriptive branch names:
- `feat/<area>-<summary>`
- `fix/<area>-<summary>`
- `docs/<area>-<summary>`
- `chore/<area>-<summary>`

Examples:
- `docs/contributor-workflow`
- `fix/dqn-target-update`

### Commit messages
Follow a conventional style:
- `feat: add replay warmup guard`
- `fix: prevent empty batch sampling`
- `docs: document canonical uv commands`
- `chore: add pre-commit hooks`

Keep the subject line imperative and under ~72 characters.

## Canonical command set (deterministic workflow)
Always run commands via `uv run`:

```bash
uv run make format
uv run make lint
uv run make typecheck
uv run make test
uv run make docs
uv run pre-commit run --all-files
```

## Test matrix
CI validates on Python 3.12 and 3.13 with:
- `black --check adeptly tests`
- `mypy adeptly tests`
- `pytest -q`

For local work, run:
- Required minimum before PR: `uv run mypy adeptly tests` and `uv run pytest -q`
- Recommended full sweep: canonical command set above.

## Pre-commit hooks
Install and run with `uv`:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

The configured hooks enforce formatting, linting, and type checks before commit.

## Pull request checklist
- [ ] Branch name follows the naming convention.
- [ ] Commits follow conventional commit style.
- [ ] `uv sync --frozen --extra dev` completed successfully.
- [ ] `uv run make lint` passes.
- [ ] `uv run make typecheck` passes.
- [ ] `uv run make test` passes.
- [ ] `uv run pre-commit run --all-files` passes.
- [ ] Documentation updated for behavior/workflow changes.
- [ ] PR description includes motivation, scope, and validation results.
