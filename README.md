# NHSMM Interfaces

Domain-oriented interface definitions and integration contracts for Neural Hidden Semi-Markov Models (NHSMM).

This repository contains interface and adapter contracts intended to separate domain-specific integration code from the NHSMM core package.

## Relationship to NHSMM

- **Core modeling and inference:** [awa-si/nhsmm](https://github.com/awa-si/nhsmm)
- **Integration contracts:** `nhsmm-interfaces`

`nhsmm-interfaces` does not implement the NHSMM model itself and does not define domain-specific decision logic. It provides boundaries, data contracts, and adapter-oriented interfaces for systems that integrate with NHSMM outputs or inputs.

Conceptually:

```text
domain system
    |
engine binding
    |
UniversalAdapter / NHSMMRuntimeAdapter
    |
nhsmm core/runtime
```

## Adapters

The adapter layer provides a common integration boundary for host frameworks such as Nautilus Trader, Freqtrade, and other event-driven or batch systems.

The canonical streaming path is:

```text
host event
  -> to_observation(event)
  -> to_context(event)
  -> NHSMMRuntimeAdapter.infer(...)
  -> nhsmm.HSMMFilterRuntime
  -> StateEstimate
  -> from_state(state)
  -> host-facing result
```

Public contracts:

- `Observation`: ordered model feature values plus optional timestamp, instrument, and metadata;
- `Context`: optional external model-context values plus metadata;
- `StateEstimate`: most-probable latent state, state posterior, episode-age posterior, timestamp, and metadata;
- `UniversalAdapter`: engine-neutral orchestration contract;
- `NHSMMRuntimeAdapter`: bridge from the canonical contracts to the public NHSMM streaming runtime.

Concrete framework adapters should normally subclass `NHSMMRuntimeAdapter` and implement only host-specific translation such as `to_observation()` and, when external context is used, `to_context()`.

They should not contain strategy rules, signal generation, portfolio policy, execution logic, or risk policy.

See **[`docs/adapters.md`](docs/adapters.md)** for the full adapter architecture, lifecycle rules, minimal usage example, internal/external context handling, and guidance for Nautilus/Freqtrade integrations.

## Scope

The repository is intended to cover contracts such as:

- sequence and observation inputs;
- context and metadata inputs;
- latent-state and posterior outputs;
- duration and transition reporting;
- batch and streaming adapter boundaries;
- domain-facing normalization of NHSMM results.

The interface layer is kept separate from the probabilistic model so that integration code does not depend directly on NHSMM internals where a narrower contract is sufficient.

## Interface groups

Current domain groupings include:

### Security and cyber-physical systems

Event, telemetry, streaming, and state-output contracts.

### Finance and trading

Market-data inputs and regime/state output contracts.

### IoT and industrial systems

Sensor-sequence inputs and operational-state outputs.

### Healthcare and clinical-research access

AWA Access healthcare/research workflow contracts. `AWAAccessResearchAdapter` maps structured AWA Access worker/API events into NHSMM observations/context while preserving case, document, assessment, and AI-processing metadata; see [`docs/adapters.md`](docs/adapters.md#awa-access-healthcare-and-clinical-research).

### Robotics and motion analytics

Motion-sequence inputs and temporal state outputs.

### Telecommunications and network analytics

Network/flow sequence inputs and state-reporting contracts.

### Energy and grid systems

Telemetry inputs and state/transition reporting contracts.

### Generic and research interfaces

Domain-neutral sequence containers, posterior access, and evaluation hooks.

## Design boundaries

The repository should remain focused on interface contracts rather than model implementation.

In particular:

- NHSMM inference and training belong in [`awa-si/nhsmm`](https://github.com/awa-si/nhsmm).
- Domain policy and application decisions belong in downstream systems.
- Interfaces should expose only the model information required by downstream consumers.
- Domain-specific adapters should not change NHSMM core semantics.

## Documentation

- [`docs/adapters.md`](docs/adapters.md) — adapter architecture and usage

## Status

This repository is under active development. Interface contracts may change until they are explicitly documented as stable.

## License

Apache License 2.0 © AWA.SI.
