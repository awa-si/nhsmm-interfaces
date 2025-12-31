# Interfaces for NHSMM (Neural Hidden Semi-Markov Models)

**Domain-oriented interface definitions and integration contracts for Neural Hidden Semi-Markov Models (NHSMM).**

This repository provides **standardized interfaces** that enable **domain-specific systems** to integrate with **NHSMM-based models** in a consistent, modular, and scalable manner.  
It serves as the **contract layer** between the NHSMM core library and **multi-domain applications** within the **State Aware Engine (SAE)** ecosystem.

---

## 🔗 Relationship to NHSMM

- **Core Modeling & Inference**: [NHSMM](https://github.com/awa-si/NHSMM)  
- **This Repository**: `nhsmm-interfaces` — domain-facing contracts and abstractions

`nhsmm-interfaces` does not implement domain logic or probabilistic models. It defines **stable boundaries** that allow domain systems to evolve independently of NHSMM internals.

---

## 🎯 Design Intent

- Align NHSMM integration with **real-world, multi-domain use cases**
- Decouple **domain semantics** from **latent-state modeling**
- Enable **consistent state-aware behavior** across heterogeneous systems
- Support research, production, cloud, on-prem, and edge deployments

---

## 🌐 Interface Groups (Multi-Domain Aligned)

### 1. Security & Cyber-Physical Systems Interfaces

Interfaces for **state-aware monitoring, anomaly detection, and event-driven systems**.

**Focus**
- Latent operational or threat states
- Temporal anomaly signaling
- Real-time or near-real-time inference

**Interface Scope**
- State event emission contracts
- Anomaly and regime-change signaling
- Streaming and log-based sequence adapters

---

### 2. Finance & Trading Interfaces

Interfaces for **market regime modeling and adaptive financial systems**.

**Focus**
- Regime detection and transition tracking
- Time-varying risk or strategy states
- Portfolio- or asset-level sequence abstraction

**Interface Scope**
- Time-series market data adapters
- Regime state output contracts
- Strategy-aware state transition hooks

---

### 3. IoT & Industrial Systems Interfaces

Interfaces for **sensor-driven and machine-state modeling**.

**Focus**
- Equipment operational states
- Predictive maintenance regimes
- Multi-sensor temporal fusion

**Interface Scope**
- Sensor sequence containers
- State persistence and dwell-time reporting
- Edge-friendly inference boundaries

---

### 4. Health & Wearables Interfaces

Interfaces for **physiological and activity-based state modeling**.

**Focus**
- Latent activity or health states
- Personalized temporal patterns
- Multimodal, noisy time-series data

**Interface Scope**
- Wearable and biosignal adapters
- Patient- or user-centric state outputs
- Privacy-aware data exchange contracts

---

### 5. Robotics & Motion Analytics Interfaces

Interfaces for **behavioral and motion-state tracking**.

**Focus**
- Robot or agent behavior states
- Safe transition detection
- Temporal task segmentation

**Interface Scope**
- Telemetry and motion sequence adapters
- State transition alerts
- Control-system integration boundaries

---

### 6. Telecommunications & Network Analytics Interfaces

Interfaces for **network-level temporal state analysis**.

**Focus**
- Latent congestion or traffic regimes
- Temporal anomaly detection
- High-throughput sequential data

**Interface Scope**
- Network flow sequence adapters
- Regime-change signaling
- Scalable batch and streaming interfaces

---

### 7. Energy & Smart Grid Interfaces

Interfaces for **state-aware energy system monitoring**.

**Focus**
- Load, failure, or stability regimes
- Long-horizon temporal dependencies
- Infrastructure-scale sequences

**Interface Scope**
- Grid telemetry adapters
- State persistence and transition reporting
- Planning and optimization hooks

---

### 8. Cross-Domain Research & AI Interfaces

Interfaces for **experimental and hybrid sequence modeling**.

**Focus**
- Novel HSMM/HMM variants
- Research-driven extensions
- Multi-domain abstraction reuse

**Interface Scope**
- Generic sequence containers
- Inference and posterior access contracts
- Experimentation and evaluation hooks

---

## 🧩 Role in the State Aware Engine (SAE)

Within SAE, `nhsmm-interfaces` enables:

- **Consistent state semantics** across domains  
- **Pluggable adapters** without modifying NHSMM core  
- **Clean separation** between probabilistic modeling and application logic  

This allows SAE to scale horizontally across industries while maintaining a unified temporal modeling foundation.

---

## 📄 License

Released under the **Apache License 2.0** © 2025 **AWA.SI**
