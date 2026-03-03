# Contributing

## Setup

```bash
uv sync --group dev
```

## Testing

```bash
uv run pytest                    # full suite (requires GPU adapter)
uv run pytest -m "not gpu"       # skip GPU tests
uv run pytest --cov              # with coverage
```

## Code Quality

```bash
uv run ruff check .              # lint
uv run ruff format .             # format
uv run ruff format --check .     # verify formatting
```

Lint rules enforced: E, F, W, I, UP, B. Max line length: 120.

## Pull Requests

1. Fork and create a feature branch
2. Make changes
3. Pass all tests and linting locally
4. Open PR against `main`
5. Describe *why* the change was made, not just what changed
