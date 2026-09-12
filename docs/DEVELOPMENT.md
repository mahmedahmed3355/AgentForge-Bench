# Development

## Environment

AgentForge-Bench targets modern Python and Prime Verifiers V1.

Use uv for environment and dependency management.

## Basic Checks

Install the project:

uv sync

Run tests:

uv run pytest

Verify the package:

uv run python -c "import agentforge; print(agentforge.__version__)"

## Development Rule

Do not add placeholder files or architecture merely for appearance.

Every environment must contain a real task specification, state model, interaction mechanism, scoring/verifier logic, tests, and reproducible evaluation behavior.
