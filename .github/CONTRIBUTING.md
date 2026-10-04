# Contributing to NHSMM Interfaces

`nhsmm-interfaces` is the integration and orchestration layer around the public `awa-si/nhsmm` API. Changes should preserve repository boundaries: model semantics remain in core, framework mapping/orchestration belongs here, and downstream decision policy stays outside.

## Development setup

This repository is currently source-deployed and does not yet define one unified package manifest for all adapters. Make the repository root importable and install the dependencies required by the slice you are changing.

For generic adapter/evaluator work, install a compatible public `nhsmm` package and PyTorch. Nautilus-specific tests additionally require a compatible NautilusTrader installation.

The current core contract requires Python 3.12+.

## Before opening a pull request

Run the smallest test set that covers the change. Run all available relevant tests when shared adapter contracts, public exports, walk-forward orchestration, or framework lifecycle behavior changes.

Examples:

```bash
python -m pytest -q tests/test_evaluation.py
python -m pytest -q tests --ignore=tests/test_nautilus_adapter.py
```

Run the Nautilus adapter tests when the required NautilusTrader dependency is installed.

Only report checks that were actually executed. State explicitly when a dependency prevented a test from running.

## Boundary review

Review each change for the correct ownership layer.

Belongs here:

- host/framework object mapping;
- canonical observation/context/state contracts;
- runtime lifecycle wiring;
- strict chronological walk-forward orchestration;
- composition of public NHSMM configuration, tuning, health, and validation APIs.

Does not belong here:

- NHSMM model/optimizer implementation;
- private NHSMM internals;
- trading signals or PnL objectives;
- portfolio/risk policy;
- order/execution policy;
- unrelated application business logic.

## Compatibility

When changing public adapter/evaluator contracts, update source, tests, README, and detailed documentation together. Do not silently reorder features, reinterpret timestamps, alter posterior semantics, or weaken train/OOS ordering guarantees.

## Documentation

Document implemented behavior rather than intended future behavior. Keep root README concise; detailed runtime and evaluation contracts belong under `docs/` or the concrete framework adapter directory.

## Pull requests

Keep pull requests focused. Include:

- what changed;
- which interface or lifecycle contract is affected;
- compatibility implications;
- exact verification performed;
- unavailable checks or unresolved assumptions.

## License

Contributions are licensed under the repository's Apache-2.0 [`LICENSE`](../LICENSE).
