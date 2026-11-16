"""Minimal evaluation harness for the agent.

Reading order: `records` (data types) -> `checks` (pure verdicts) -> `suite` (case schema and
loader) -> `arxiv_fake` (per-case offline arXiv) -> `runner` (one run -> one record) ->
`store` (SQLite) -> `report` (aggregation, markdown) -> `run` (CLI).

    uv run python -m evals.run --suite minimal --repeat 3 --label baseline
    uv run python -m evals.report --label baseline [--compare other_label]
"""
