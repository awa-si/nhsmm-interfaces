# NHSMM Interfaces — Research & Early Access

**Domain-oriented interface definitions and integration contracts for Neural Hidden Semi-Markov Models (NHSMM).**

This repository provides **standardized interfaces** enabling **domain-specific systems** to integrate with **NHSMM-based models** in a consistent, modular, and scalable way.  
It serves as the **contract layer** between the NHSMM core library and **multi-domain applications** within the **State Aware Engine (SAE)** ecosystem.

> ⚠️ This repository is currently in **research preview / early access**. After initial showcase releases, it may become closed-access. Full access and research updates are available to Patreon supporters. See [Patreon Tier Details](#patreon-early-access).

---

## 🔗 Relationship to NHSMM & SAE

- **Core Modeling & Inference**: [NHSMM](https://github.com/awa-si/NHSMM) — fully open-source and actively developed  
- **Interfaces / Contract Layer**: `nhsmm-interfaces` — early access for research, experimentation, and SAE preparation  
- **SAE**: Planned commercial/research product using `nhsmm-interfaces` as the foundation

`nhsmm-interfaces` **does not implement domain logic**; it defines **stable boundaries** allowing domain systems to evolve independently of NHSMM internals.

---

## 🎯 Design Intent

- Align NHSMM integration with **real-world, multi-domain use cases**  
- Decouple **domain semantics** from **latent-state modeling**  
- Enable **consistent state-aware behavior** across heterogeneous systems  
- Support research, production, cloud, on-prem, and edge deployments  
- Offer early access to **researchers and subscribers** via Patreon

---

## 🌐 Interface Groups (Multi-Domain Aligned)

### 1. Security & Cyber-Physical Systems
- **Focus**: Latent operational or threat states, anomaly signaling, real-time inference  
- **Scope**: Event emission contracts, streaming/log-based adapters

### 2. Finance & Trading
- **Focus**: Market regime detection, time-varying strategies, portfolio states  
- **Scope**: Market data adapters, regime output contracts

### 3. IoT & Industrial Systems
- **Focus**: Sensor-driven operational states, predictive maintenance  
- **Scope**: Sensor sequence containers, dwell-time reporting

### 4. Health & Wearables
- **Focus**: Latent activity/health states, multimodal time-series  
- **Scope**: Wearable adapters, patient-centric state outputs

### 5. Robotics & Motion Analytics
- **Focus**: Robot/agent behavior states, temporal task segmentation  
- **Scope**: Motion sequence adapters, control-system boundaries

### 6. Telecommunications & Network Analytics
- **Focus**: Network traffic regimes, anomaly detection  
- **Scope**: Flow sequence adapters, scalable batch/streaming interfaces

### 7. Energy & Smart Grid
- **Focus**: Load/failure regimes, long-horizon dependencies  
- **Scope**: Grid telemetry adapters, transition reporting

### 8. Cross-Domain Research & AI
- **Focus**: Experimental HSMM/HMM variants, multi-domain abstraction  
- **Scope**: Generic sequence containers, posterior access, evaluation hooks

---

## 🧩 Role in SAE

`nhsmm-interfaces` enables:

- **Consistent state semantics** across domains  
- **Pluggable adapters** without modifying NHSMM core  
- **Clean separation** of probabilistic modeling and application logic  

This allows SAE to scale horizontally across industries while maintaining a **unified temporal modeling foundation**.

---

## 💡 Patreon Early Access

Support the research and early interface development on **Patreon**:

- **Research Preview Tier**: Download per-release snapshots, access research sketches  
- **Insider Tier**: Unlimited interface access, architectural insights, priority discussions  
- **SAE Founders Tier**: Early SAE product access and roadmap influence  

> Supporting Patreon helps fund ongoing NHSMM research and accelerates SAE development.

---

## 📄 License

Released under the **Apache License 2.0** © 2025 **AWA.SI**
