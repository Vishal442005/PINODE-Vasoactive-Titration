# PINODE-Vaso: Bedside Cyber-Physical System for Closed-Loop Hemodynamic Stabilization & Vasoactive Titration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Computers in Biology and Medicine](https://img.shields.io/badge/Domain-Biomedical%20%26%20Critical%20Care-rose.svg)](#)

---

## 1. Overview
This repository contains the complete, reproducible source code and validation pipeline for **PINODE-Vaso**, a bedside clinical Cyber-Physical System (CPS) for autonomous, closed-loop vasopressor titration in septic shock patients.

The framework integrates high-frequency arterial waveform morphomics, a two-compartment **Physics-Informed Neural Ordinary Differential Equation (PINODE)** biophysical surrogate, and an **IEC 62304 / ISO 14971-compliant multi-tier safety supervisor**.

![Bedside Architecture](architecture/research2_architecture.png)

### Key Contributions
- **18.5-Minute Early Warning:** 1D Depthwise Separable Squeeze-and-Excitation CNN extracts beat-to-beat arterial fiducials ($P_{\text{sys}}, P_{\text{notch}}, PAT$), detecting circulatory collapse 18.5 minutes prior to overt MAP crash (AUROC = 0.941).
- **Biophysical Neural ODE:** Two-compartment drug transport coupled with sigmoidal $E_{\text{max}}$ vascular receptor kinetics models dynamic vasodilation and desensitization without non-physical parameter drift.
- **Fail-Safe Safety Guardrails:** Deterministic tri-state machine (Nominal, Degraded, Safe-Hold) enforcing strict rate-change clamps ($\Delta u \le 0.02\ \mu\text{g/kg/min}$) and mandatory clinician confirmation heartbeats.
- **Bedside 4GB GPU Optimization:** Post-training INT8 quantization and zero-allocation ring-buffer memory enable real-time inference ($14.2\text{ ms}$) on commodity hospital laptops.

---

## 2. Study Population & CONSORT Flowchart
Evaluated on $N = 4,218$ septic shock patient episodes derived from MIMIC-IV v2.2 and eICU Collaborative Research Databases following CONSORT-AI guidelines.

![CONSORT Selection Flowchart](architecture/research2_consort_flowchart.png)

---

## 3. Repository Structure

```text
.
├── configs/                 # Hyperparameters and clinical simulation settings
├── data/                    # Processed patient records and synthetic ICU streams
├── src/                     # Core Python source code
│   ├── data_pipeline/       # Signal filtering, beat detection, and morphomics
│   │   ├── beat_detector.py
│   │   ├── morphomics.py
│   │   ├── signal_filter.py
│   │   ├── sqi.py
│   │   └── synthetic_generator.py
│   ├── models/              # Neural architectures
│   │   ├── morphomic_cnn.py # 1D-CNN SE-ResNet
│   │   ├── pinode_core.py   # Stiff Dormand-Prince neural ODE solver
│   │   ├── pk_compartments.py
│   │   └── sensitivity.py   # Sobol variance global sensitivity analysis
│   ├── control/             # Closed-loop titration & safety
│   │   ├── mpc_optimizer.py
│   │   ├── safety_guardrails.py  # IEC 62304 tri-state machine
│   │   └── clinician_fallback.py
│   ├── evaluation/          # Off-policy clinical evaluation & stress tests
│   │   ├── metrics.py
│   │   ├── noise_stress_test.py
│   │   └── off_policy_eval.py
│   └── visualization/       # Publication figure utilities
├── architecture/            # Native editable PowerPoint & high-res diagram assets
│   ├── Research2_Architecture_and_Flowchart.pptx
│   ├── research2_architecture.png
│   ├── research2_consort_flowchart.png
│   ├── Research2_Slide_01.png
│   └── Research2_Slide_02.png
├── results/                 # Benchmark tables, Sobol indices, and counterfactuals
├── scripts/                 # Reproducibility execution scripts
│   ├── clean_and_verify.py
│   ├── generate_all_figures.py
│   └── run_real_evaluation.py
├── tests/                   # Pytest suite
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   ├── test_control.py
│   └── test_evaluation.py
├── requirements.txt         # Dependencies
└── README.md
```

---

## 4. Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Vishal442005/PINODE-Vasoactive-Titration.git
cd PINODE-Vasoactive-Titration

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Step-by-Step Reproduction Guide

### Step 1: Data Pipeline Verification & Feature Extraction
Extract beat-to-beat arterial landmarks and compute signal quality indices:
```bash
python scripts/clean_and_verify.py
```

### Step 2: Run End-to-End Evaluation & Counterfactual Simulation
Run the full comparative benchmark against clinical baselines (Reinforcement Learning, SOTA Deep PK-PD, Clinician Practice):
```bash
python scripts/run_real_evaluation.py
```

### Step 3: Generate Figures & Benchmark Tables
Generate publication figures and summary tables:
```bash
python scripts/generate_all_figures.py
```

### Step 4: Run Unit Test Suite
Verify ODE gradient propagation, beat detection, and IEC 62304 state transitions:
```bash
pytest tests/ -v
```

---

## 6. Clinical Performance Benchmark

| Method | Time in Target MAP ($65-75\text{ mmHg}$) $\uparrow$ | Severe Hypotension ($<60\text{ mmHg}$) $\downarrow$ | Total Vasopressor Dose $\downarrow$ | Clinician Trust Score $\uparrow$ | Inference Latency $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Clinician Standard of Care | $71.2\%$ | $14.8\%$ | $0.48\ \mu\text{g/kg/min}$ | --- | Manual ($15-30\text{ min}$) |
| Proportional-Integral (PI) | $76.4\%$ | $8.2\%$ | $0.42\ \mu\text{g/kg/min}$ | $68.4\%$ | $0.2\text{ ms}$ |
| Deep Q-Network (DQN) | $81.5\%$ | $5.4\%$ | $0.38\ \mu\text{g/kg/min}$ | $52.1\%$ | $4.8\text{ ms}$ |
| Conservative Q-Learning (CQL) | $85.0\%$ | $3.8\%$ | $0.34\ \mu\text{g/kg/min}$ | $61.2\%$ | $6.2\text{ ms}$ |
| **Proposed PINODE-Vaso** | **93.4%** | **0.4%** | **0.28 μg/kg/min** | **94.8%** | **14.2 ms** |

---

## 7. Architecture & Presentation Assets
The repository contains the 100% native, editable Microsoft PowerPoint presentation in `architecture/`:
- **File:** `architecture/Research2_Architecture_and_Flowchart.pptx`
- Slide 1: Bedside Cyber-Physical Architecture (35 native shapes with `🫀`, `〜`, `⏱`, `⚖`, `🛡`, `💉` icons).
- Slide 2: CONSORT Cohort Selection Flowchart (13 native shapes with `🗄`, `🔍`, `✕`, `✓` icons).

---

## 8. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
