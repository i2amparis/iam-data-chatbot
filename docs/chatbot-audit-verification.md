# Chatbot audit fixes — 2026-09-17

## Study catalogue

Source: https://iamparis.eu/results. The page's server-rendered workspace
metadata exposes `internal_name`, `title`, study slug, and project slug.
All 21 local catalogue entries were checked against those fields. All 42
explorer/explainer URLs returned HTTP 200 at the expected URL.

Corrections:

- `study-3`: Tech-constrained EU pathways to net zero.
- `study-7`: Behavioural changes for climate policy.
- `decipher`: the public code for Decipher modelling studies; the previous
  `decipher_1` catalogue entry was not present in the authoritative catalogue.
  Legacy ingestion workspace identifiers were not renamed or equated to public
  codes without evidence.

To repeat the verification using a downloaded copy of the public results HTML:

```sh
.venv/bin/python verify_workspace_catalog.py /path/to/results.html --check-links
```

The verifier reads embedded metadata as text without executing website scripts.
It exits nonzero for mismatches, missing codes, redirects, and failed URLs.

## Local test environment

The project `.venv` uses the existing Python installation's site packages and
has the declared `pycountry` dependency installed locally. Run tests with this
interpreter; the system interpreter does not acquire dependencies from `.venv`.

```sh
.venv/bin/python -m unittest discover -v
```

Final result: **654 tests passed**, zero failures and zero errors. Dependency
consistency (`pip check`), compilation, and whitespace checks also passed.
Quality-gate unit tests intentionally print simulated passing and failing gate
messages; the enclosing test run returned exit status 0.

## Behaviour covered

- A newly named model overrides the previous model in qualitative follow-ups.
- Unsupported model coverage directs users to model documentation.
- Oil-demand answers disclose when the available indicator covers all liquid fuels.
- Explicit plot requests link to the explorer and preserve the last data scope.
- Comparisons use numeric tables, including through the real data handler.
- Legacy pending plot choices migrate to numeric-data handling.
- Scenario comparison tests check both selected scenarios and reject other
  scenarios and records from another study.
- Existing scope, year-filter, model-version, and routing assertions remain in
  the updated suite. Assertions for removed chart output and old wording were
  replaced with the new response contract.

Standalone plotting utilities retain their own tests; the chatbot data agent
disables that legacy rendering path.
