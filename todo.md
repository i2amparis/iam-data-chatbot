# TODO - IAM PARIS Chatbot

The backend is functionally complete: shared `RuntimeContext`, link catalog/router,
session state, availability-matrix validation, English query normalization,
deterministic routing, grounded answers, evals, monitoring and a CI quality gate
(`python quality_gate.py`). All unit tests pass.

Only the items below remain — they require the frontend, not backend work.

## Remaining

- [ ] Run actual frontend QA with the real UI once the frontend integration is available.
- [ ] Build an admin feedback dashboard UI for feedback candidates and monitoring alerts.
