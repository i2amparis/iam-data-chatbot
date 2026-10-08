# TODO - IAM PARIS Chatbot

Backend status: core pipeline works (shared `RuntimeContext`, link catalog/router,
session state, deterministic routing, grounded answers, evals, monitoring, CI
quality gate via `python quality_gate.py`). All unit tests pass.

Open response-quality, latency and tooling issues from the 2026-10-07 review
are tracked in [response_fixes_todo.md](response_fixes_todo.md).

## Remaining

- [ ] Run actual frontend QA with the real UI once the frontend integration is available.
- [ ] Build an admin feedback dashboard UI for feedback candidates and monitoring alerts.
- [ ] Work through the open items in [response_fixes_todo.md](response_fixes_todo.md).
