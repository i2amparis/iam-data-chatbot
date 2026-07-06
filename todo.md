# TODO - IAM PARIS Chatbot

Backend status: core pipeline works (shared `RuntimeContext`, link catalog/router,
session state, deterministic routing, grounded answers, evals, monitoring, CI
quality gate via `python quality_gate.py`). All unit tests pass.

Known weaknesses and their fix status are tracked in
[weaknesses_todo.md](weaknesses_todo.md) — most critical and high-priority items
are fixed; a few architectural improvements remain open there.

## Remaining

- [ ] Run actual frontend QA with the real UI once the frontend integration is available.
- [ ] Build an admin feedback dashboard UI for feedback candidates and monitoring alerts.
- [ ] Work through the remaining open items in [weaknesses_todo.md](weaknesses_todo.md).
