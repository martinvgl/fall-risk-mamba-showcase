<div align="center">

# Fall Risk as a Latent Dynamical State

### Skeleton-Based Continuous Fall Risk Estimation with State-Space Models

*A master's research project in assistive robotics.*

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)

</div>

---

## TL;DR

Most fall detection systems answer the wrong question. They ask *"is this frame a fall?"* and return a flickering binary signal — useless for a robot that needs to decide, continuously, how much to assist a human.

This project reformulates the problem: instead of classifying frames, it **estimates a continuous fall-risk signal `r(t) ∈ [0, 1]`** from 2D skeleton keypoints, using a Directed Graph Neural Network (DGNN) and a Mamba state-space model. The output is a smooth, anticipatory signal that can feed directly into a robotic control policy.

> 🖼️ *`[figure: risk curve vs binary classifier output on a fall sequence]`*

---

## What This Project Demonstrates

This is a master's research project, but it showcases engineering and ML skills relevant to industry:

- **End-to-end ML pipeline** — from raw video to trained model, with YOLO-pose keypoint extraction, signal processing (Butterworth, interpolation), PyTorch training, and evaluation.
- **Multi-dataset benchmarking** — nine public fall detection datasets (URFD + 8 from OmniFall) unified under a single preprocessing pipeline driven by YAML config.
- **Rigorous evaluation** — multi-seed runs, confidence intervals, McNemar statistical tests, per-dataset metrics. No cherry-picking.
- **Modern architectures** — Mamba state-space models, Directed Graph Neural Networks, causal Transformers. Not just off-the-shelf.
- **Bridging control theory and deep learning** — explicit motivation for SSMs grounded in the analogy between their state update and classical control systems (`ẋ = Ax + Bu`).
- **Reproducibility** — fixed seeds, versioned configs, documented environment, CLI-driven pipeline.

---

## Architecture

```
Input: (batch, T=30, 13, 2) — windowed skeleton sequence
           ↓
    DGNN  (per-frame spatial encoding)
    4 layers, output: g_t ∈ R^128 per frame
           ↓
    Velocity concatenation: [g_t, Δg_t] ∈ R^256
           ↓
    Mamba SSM  (temporal latent state dynamics)
    2 layers, d_model=256, d_state=64
    State update: x_{t+1} = A(u_t) x_t + B(u_t) u_t
           ↓
        Output head
        /         \
   Classifier    Risk head
       ↓             ↓
   {S, T, F}    r_t ∈ [0, 1]
```

> 🖼️ *`[figure: full pipeline diagram, from raw video to r(t) output]`*

**Why Mamba over LSTM?** LSTM has implicit recurrence with no structural constraint on state continuity. Mamba explicitly parameterizes state-space dynamics, providing an architectural inductive bias toward smooth latent trajectories — exactly what a control system requires.

**Why velocity features?** Balance stability in biomechanics depends on both position and velocity (Hof et al., 2005 — Margin of Stability). Feeding `[g_t, Δg_t]` aligns the input with the SSM's own recurrence structure.

---

## Selected Results

All results on URFD, averaged over 5 seeds (see full README in repo for details).

| Model              | Detection rate | False alarm rate | Notes                               |
|--------------------|----------------|------------------|-------------------------------------|
| DGNN + LSTM        | 93.7%          | 21.7%            | Cannot reduce FA below 21.7%        |
| DGNN + Transformer | 61.1%          | 3.9%             | Step-function threshold behavior    |
| **DGNN + Mamba**   | **67.4%**      | **14.7%**        | Tunable down to 7.0% FA at τ = 0.50 |

**Velocity ablation** — removing `Δg_t`:
- Mamba degrades by ~1.7×
- LSTM degrades by ~8×

This gap reflects Mamba's inductive bias: its state update implicitly models continuous-time dynamics, so it relies less on hand-crafted velocity features.

> 🖼️ *`[figure: confusion matrices or risk curve comparisons across the three models]`*

Statistical significance: LSTM vs. Mamba `p < 0.001` (McNemar). Transformer vs. Mamba: not significant.

**Cross-dataset generalization on OmniFall-Staged (8 datasets): in progress.**

---

## Benchmarks Used

The project unifies **nine** public fall detection datasets under one preprocessing pipeline:

URFD · Le2i · GMDCSA-24 · CAUCAFall · UP-Fall · EDF · OCCU · MCFD · CMDFall

Each dataset has its own quirks (frame rate, naming conventions, occlusions, camera angles, compression). The pipeline handles them uniformly via a single YAML config.

> 🖼️ *`[figure: sample frames from each of the 9 datasets, showing diversity]`*

---

## Tech Stack

**Core**
- Python 3.8+, PyTorch 2.0+, CUDA 12.1
- [Mamba-SSM](https://github.com/state-spaces/mamba) (selective state-space models)
- [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) (pose estimation)

**Data & Signal Processing**
- NumPy, SciPy (Butterworth filtering, linear interpolation)
- OpenCV, FFmpeg (video I/O and conversion)
- HuggingFace `datasets` (OmniFall annotation loading)

**ML & Evaluation**
- scikit-learn (metrics, McNemar tests)
- Matplotlib, Seaborn (visualization)
- PyYAML (config management)

**Development**
- WSL2 Ubuntu on Windows
- Git / GitHub
- CUDA-enabled GPU (training and inference)

---

## Project Structure

```
fall-risk-mamba/
├── configs/                     # YAML configs (datasets, model, training)
├── data/
│   ├── raw/<DATASET>/           # Raw videos, one subfolder per dataset
│   ├── processed/<DATASET>/     # Extracted keypoints, cleaned, normalized, 30fps
│   └── annotations/             # Frame-level labels from OmniFall
├── preprocessing/               # Dataset-agnostic pipeline scripts
├── models/                      # DGNN, Mamba, LSTM, Transformer architectures
├── training/                    # Unified training entry point
├── evaluation/                  # Metrics, evaluation, visualization
└── results/                     # Checkpoints, figures, metrics (.json)
```

The preprocessing pipeline is **dataset-agnostic**: one YAML file, one CLI flag (`--dataset <name>`), and the same four scripts handle video, image sequences, or zipped archives across nine datasets.

---

## Quick Start

```bash
git clone https://github.com/martinvgl/fall-risk-mamba.git
cd fall-risk-mamba

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install mamba-ssm  # requires CUDA

# Preprocess any configured dataset
python preprocessing/extract_keypoints.py   --dataset Le2i
python preprocessing/clean_keypoints.py     --dataset Le2i
python preprocessing/normalize_keypoints.py --dataset Le2i
python preprocessing/resample_to_30fps.py   --dataset Le2i

# Train
python training/train.py --model mamba --objective risk --epochs 100

# Evaluate
python evaluation/evaluate.py --checkpoint results/checkpoints/mamba_risk_best.pth \
                              --model mamba --objective risk
```

See the [detailed README](./README.md) for full usage, preprocessing for all 9 datasets, and evaluation options.

---

## What I Built This With

This project started from a simple observation: a robot cannot stabilize an unsteady person based on flickering binary labels. That led me into control theory analogies (Laplace transforms, state-space representations from my Polytech Nancy background), which in turn led to state-space models as a natural fit for continuous risk estimation.

Along the way I had to:

- **Debug training variance** on a small dataset (59 training videos) — traced back to unfixed random seeds and scheduler mismatches, solved with systematic multi-seed evaluation.
- **Pivot the benchmark** when single-dataset results proved unreliable — moved to the OmniFall multi-domain benchmark (Feb 2026 release).
- **Reconcile conflicting labels** — URFD's "transient" means *"is falling"*, not *"pre-fall instability"*. Getting this wrong would have invalidated the entire framing.
- **Build infrastructure** — a unified preprocessing pipeline that replaced 10+ hardcoded scripts with 5 generic ones driven by config.

Most of the work was not the model. Most of the work was making the problem tractable.

---

## About Me

**Martin Vogel** · Master's student in engineering · Polytech Nancy × Kyushu Institute of Technology

Background in industrial automation (PLC, SCADA, fieldbus, control theory), pivoted toward robotics and applied ML. This project is part of my master's research under Prof. Tomohiro Shibata in Fukuoka, Japan. I'm looking to join a robotics-focused engineering team in France after graduation.

📫 **Contact**: vogel.martin-romain7334@mail.kyutech.jp

---

## Key References

- **Schneider et al. (2025)** — *OmniFall: A Unified Multi-Domain Dataset for Robust Fall Detection*, arXiv:2505.19889
- **Gu & Dao (2023)** — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752
- **Shi et al. (2019)** — *Skeleton-Based Action Recognition with Directed Graph Neural Networks*, CVPR 2019
- **Hof et al. (2005)** — *The Condition for Dynamic Stability*, J. Biomechanics

Full bibliography in the detailed [README](./README.md).

---

<div align="center">

*🚧 Work in progress — master's thesis expected September 2026*

</div>
