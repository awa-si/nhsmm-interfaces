# Support

Use GitHub Issues for reproducible defects or concrete interface changes in `nhsmm-interfaces`.

Before opening an issue:

- check the root README and relevant documentation;
- identify whether the issue belongs to `nhsmm`, `nhsmm-interfaces`, NautilusTrader, or a downstream application;
- reduce the report to the smallest reproducible adapter/evaluator case;
- include exact versions or commit SHAs where possible.

Repository ownership boundaries:

- NHSMM model, artifacts, inference, filtering, forecasting, configuration, and core validation: `awa-si/nhsmm`;
- framework adapters, lifecycle integration, and walk-forward orchestration: this repository;
- strategy, signal, risk, portfolio, execution, or other application policy: downstream project.

For security-sensitive reports, follow [SECURITY.md](SECURITY.md) instead of filing a public issue.

General support is best-effort; no response-time SLA is implied.
