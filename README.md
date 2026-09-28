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
UniversalAdapter
    |
nhsmm core/runtime
```

## Universal adapter contract

`adapters.UniversalAdapter` is the common integration boundary for host frameworks such as Nautilus Trader, Freqtrade, or other event-driven/batch systems.

The canonical pipeline is:

```text
host event
  -> to_observation(event)
  -> to_context(event)
  -> infer(observation, context)
  -> from_state(state)
  -> host-facing result
```

Canonical data contracts:

- `Observation`: model feature values plus optional timestamp, instrument, and metadata;
- `Context`: optional model context values plus metadata;
- `StateEstimate`: latent state, posterior probabilities, optional duration, timestamp, and metadata;
- `UniversalAdapter`: orchestration contract connecting those types.

Concrete engine adapters should be thin translations around this contract. They should not contain strategy rules, signal generation, portfolio policy, execution logic, or risk policy.

Examples of engine-specific bindings that can implement this contract:

```text
NautilusAdapter   -> UniversalAdapter
FreqtradeAdapter  -> UniversalAdapter
OtherAdapter      -> UniversalAdapter
```

The engine binding owns host-object conversion and lifecycle integration. NHSMM model semantics remain in the core package.

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

### Health and wearables

Time-series and multimodal observation contracts.

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

## Status

This repository is under active development. Interface contracts may change until they are explicitly documented as stable.

## License

Apache License 2.0 © AWA.SI.
