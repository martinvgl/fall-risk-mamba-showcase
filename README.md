![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/status-WIP-yellow.svg)

# Fall Risk as a Latent Dynamical State for Assistive Robotics

> **Status**: 🚧 Work in Progress — Master's Research Project
> **Author**: Martin Vogel
> **Institution**: Kyushu Institute of Technology (Shibata Laboratory)
> **Supervisor**: Tomohiro Shibata
> **Last Updated**: April 2026

---

## Abstract

Standard fall detection frames the problem as frame-wise classification: *"Is this a fall?"*. This produces discrete, potentially flickering decisions that are poorly suited for robotic control loops.

This work reformulates fall risk as a **continuously evolving latent state**, modeled with a Directed Graph Neural Network (DGNN) for spatial encoding and a Mamba State-Space Model (SSM) for temporal dynamics. The output is a smooth risk signal `r_t ∈ [0, 1]` that can be fed directly into an assistive control policy.

**Core claim**: explicitly modeling instability as a latent state produces signals that are more stable, more anticipatory, and more compatible with assistive decision logic than direct classification.

---

## Approach

The method operates on 2D skeleton keypoints extracted with YOLO-pose from RGB video. Each frame is encoded by a DGNN capturing the directed structure of the human skeleton. Velocity features `Δg_t` are concatenated to position features `g_t` to provide the temporal model with information aligned to its own recurrence structure. A Mamba SSM then integrates the sequence into a continuous latent state, whose projection onto `[0, 1]` serves as the risk signal.

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
    State update: x_{t+1} = A(u_t)x_t + B(u_t)u_t
           ↓
         Head
        /    \
  Classifier  Risk Head
       ↓           ↓
   {S, T, F}   r_t ∈ [0, 1]
```

Two training objectives are compared: discrete classification into `{stable, transient, fall}` for comparability with prior work, and continuous risk regression as the core contribution. LSTM and causally-masked Transformer baselines are trained under the same conditions to isolate the effect of the SSM inductive bias.

---

## Datasets

All datasets are unified under the **OmniFall-Staged** benchmark (Schneider et al., 2025), which provides 16-class dense temporal annotations across eight public fall detection datasets:

| Dataset    | Type              | Videos | Segments | Duration |
|------------|-------------------|--------|----------|----------|
| CMDFall    | multi (7 views)   | 384    | 6,026    | 7.12 h   |
| UP-Fall    | multi (2 views)   | 1,118  | 1,213    | 4.59 h   |
| Le2i       | single            | 190    | 967      | 0.79 h   |
| GMDCSA24   | single            | 160    | 458      | 0.36 h   |
| CAUCAFall  | single            | 100    | 258      | 0.28 h   |
| EDF        | multi (2 views)   | 10     | 254      | 0.22 h   |
| OCCU       | multi (2 views)   | 10     | 245      | 0.25 h   |
| MCFD       | multi (8 views)   | 192    | 169      | 0.20 h   |

Additional development experiments are conducted on **URFD** (Kepski & Kwolek, 2014) as an independent held-out set for ablation studies.

---

## Project Structure

```
fall-risk-mamba/
│
├── configs/
│   ├── datasets.yaml            # Per-dataset preprocessing config (fps, input type, extensions)
│   ├── dgnn_config.yaml         # DGNN graph structure, layer sizes
│   ├── mamba_config.yaml        # Mamba SSM hyperparameters
│   └── training_config.yaml     # Loss weights, optimizer, scheduler
│
├── data/
│   ├── dataset.py               # Unified PyTorch Dataset (multi-dataset aware)
│   ├── annotations/             # Per-dataset frame-level labels (.csv)
│   ├── raw/                     # Raw videos per dataset
│   │   ├── URFD/
│   │   ├── Le2i/
│   │   ├── GMDCSA/
│   │   ├── CAUCA/
│   │   ├── UP-Fall/
│   │   ├── EDF/
│   │   ├── OCCU/
│   │   ├── MCFD/
│   │   └── CMDFall/
│   └── processed/
│       └── <DATASET>/
│           ├── keypoints/         # Raw YOLO-pose output: (T, 13, 2)
│           ├── keypoints_clean/   # Interpolation + Butterworth filter
│           ├── keypoints_norm/    # Body-relative normalization
│           ├── keypoints_30fps/   # Resampled to common 30 fps
│           └── windows/           # Final sliding windows: X_*.npy, y_*.npy
│
├── evaluation/
│   ├── evaluate.py              # Evaluation script (both training objectives)
│   ├── metrics.py               # Metric functions (lead time, flickering, etc.)
│   └── visualize.py             # Visualizations (confusion matrix, risk curves)
│
├── models/
│   ├── dgnn.py                  # Directed Graph Neural Network
│   ├── mamba_module.py          # Mamba SSM temporal module
│   ├── fall_model.py            # Complete DGNN → Mamba → Head architecture
│   └── baselines/
│       ├── lstm.py              # DGNN + LSTM baseline
│       └── transformer.py       # DGNN + Transformer baseline (causal masking)
│
├── preprocessing/
│   ├── extract_keypoints.py     # Step 1 — YOLO-pose extraction (video/images/zip)
│   ├── clean_keypoints.py       # Step 2 — NaN interpolation + Butterworth filter
│   ├── normalize_keypoints.py   # Step 3 — Body-relative coordinate normalization
│   ├── resample_to_30fps.py     # Step 4 — Temporal resampling to 30 fps
│   ├── convert_omnifall_labels.py  # OmniFall HF labels → frame-level CSV
│   └── make_windows.py          # Sliding window generation (multi-dataset)
│
├── results/
│   ├── checkpoints/             # Saved model weights (.pth)
│   ├── figures/                 # Generated plots (.png)
│   └── metrics/                 # Evaluation outputs (.json, .npy)
│
└── training/
    └── train.py                 # Unified training script (all models, all objectives)
```

---

## Installation

```bash
git clone https://github.com/martinvgl/fall-risk-mamba.git
cd fall-risk-mamba

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install mamba-ssm   # requires CUDA
```

**Key dependencies:** PyTorch 2.0+, mamba-ssm, ultralytics (YOLO26), scipy, scikit-learn, matplotlib, seaborn, datasets (HuggingFace)

---

## Usage

### Step 1 — Preprocessing

The preprocessing pipeline is dataset-agnostic and driven by `configs/datasets.yaml`. Each script operates on any configured dataset via `--dataset <NAME>`.

```bash
# Example for a single dataset
python preprocessing/extract_keypoints.py    --dataset Le2i
python preprocessing/clean_keypoints.py      --dataset Le2i
python preprocessing/normalize_keypoints.py  --dataset Le2i
python preprocessing/resample_to_30fps.py    --dataset Le2i

# All datasets in sequence
for ds in URFD Le2i GMDCSA CAUCA UP-Fall EDF OCCU MCFD CMDFall; do
    python preprocessing/extract_keypoints.py   --dataset $ds
    python preprocessing/clean_keypoints.py     --dataset $ds
    python preprocessing/normalize_keypoints.py --dataset $ds
    python preprocessing/resample_to_30fps.py   --dataset $ds
done

# Fetch OmniFall labels from HuggingFace and convert to frame-level annotations
python preprocessing/convert_omnifall_labels.py --dataset all

# Build training windows (stride_train=1, stride_test=15)
python preprocessing/make_windows.py --stride_train 1 --stride_test 15
```

### Step 2 — Training

```bash
# Classification objective (baselines + main model)
python training/train.py --model lstm        --objective cls --epochs 50 --use_class_weights
python training/train.py --model transformer --objective cls --epochs 50 --use_class_weights
python training/train.py --model mamba       --objective cls --epochs 50 --use_class_weights

# Continuous risk objective (core contribution)
python training/train.py --model mamba       --objective risk --epochs 100
python training/train.py --model lstm        --objective risk --epochs 100
python training/train.py --model transformer --objective risk --epochs 100
```

### Step 3 — Evaluation

```bash
# Classification metrics
python evaluation/evaluate.py --checkpoint results/checkpoints/mamba_cls_best.pth  --model mamba --objective cls

# Risk metrics (detection rate, false alarm rate, lead time, flickering)
python evaluation/evaluate.py --checkpoint results/checkpoints/mamba_risk_best.pth --model mamba --objective risk
```

### Step 4 — Visualization

```bash
python evaluation/visualize.py --results results/metrics/mamba_cls_results.json
python evaluation/visualize.py --results results/metrics/mamba_risk_results.json
```

---

## Results

### URFD (development / ablation)

Results averaged over 5 seeds with fixed `seed=42` reference, CosineAnnealingLR scheduler.

**Continuous risk objective — detection vs. false alarm**

| Model       | Detection rate | False alarm rate | Notes |
|-------------|----------------|------------------|-------|
| DGNN + LSTM | 93.7%          | 21.7%            | Unable to reduce FA below 21.7% at any threshold |
| DGNN + Transformer | 61.1%   | 3.9%             | Step-function threshold behavior |
| **DGNN + Mamba** | **67.4%** | **14.7%**   | Controllable down to 7.0% FA at τ=0.50 |

Statistical significance: LSTM vs. Mamba: `p < 0.001` (McNemar). Transformer vs. Mamba: not significant.

**Velocity ablation (removing Δg_t features)**

| Model       | Performance degradation |
|-------------|-------------------------|
| DGNN + Mamba | ~1.7× |
| DGNN + LSTM  | ~8× |

The Mamba architecture shows substantially lower reliance on explicit velocity features, consistent with its inductive bias toward continuous-time dynamical systems.

### OmniFall-Staged

*Experiments in progress. Cross-dataset evaluation will follow the OmniFall cross-subject (CS) and cross-view (CV) splits, with per-dataset breakdown and comparison against the I3D / VideoMAE baselines reported in Schneider et al. (2025).*

---

## Key Design Decisions

**Why velocity features `Δg_t`?**
In biomechanics, balance stability depends on both position and velocity (Hof et al., 2005 — Margin of Stability). The concatenation `[g_t, Δg_t]` gives the SSM information aligned with its own recurrence structure `dx/dt = f(x, u)`.

**Why Mamba over LSTM?**
LSTM uses implicit recurrence with no structural constraint on state continuity. Mamba explicitly parameterizes state-space dynamics `x_{t+1} = A(u_t)x_t + B(u_t)u_t`, providing an architectural inductive bias toward smooth, physically consistent latent trajectories — which is exactly what a control system requires.

**Why not smooth the classification output?**
Smoothing is post-processing. The SSM constraint is model-level: it forces continuity from the inside rather than patching it from the outside.

**Why a unified multi-dataset benchmark?**
Single-dataset results in fall detection are known to overfit to scene geometry, lighting, and actor behavior. The OmniFall-Staged benchmark provides dense 16-class annotations across eight datasets under consistent cross-subject and cross-view splits, enabling a realistic measurement of generalization.

---

## References

### Benchmark

- **Schneider et al. (2025)** — *OmniFall: From Staged Through Synthetic to Wild, A Unified Multi-Domain Dataset for Robust Fall Detection*, arXiv:2505.19889

### Fall Detection — Skeleton- and SSM-Based Methods

- **Shi et al. (2019)** — *Skeleton-Based Action Recognition with Directed Graph Neural Networks*, CVPR 2019
- **Cho et al. (2025a)** — *Anticipatory Fall Detection with DGNN + LSTM*, RO-MAN 2025, IEEE, pp. 2239–2245
- **Cho et al. (2025b)** — *Wi-Fi-Based Human Fall and Activity Recognition using Transformer and DGNN*
- **Yu et al. (2025)** — *Real-Time Skeleton-Based Fall Detection using TCN + Transformer (TCNTE)*
- **Zhang et al. (2025)** — *Fall-Mamba: A Multimodal Fusion and Masked Mamba-Based Approach for Fall Detection*

### Core Architectures

- **Vaswani et al. (2017)** — *Attention Is All You Need*, NeurIPS 2017
- **Gilmer et al. (2017)** — *Neural Message Passing for Quantum Chemistry*, ICML 2017
- **Yan et al. (2018)** — *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*, AAAI 2018
- **Shi et al. (2018)** — *Non-Local Graph Convolutional Networks for Skeleton-Based Action Recognition*
- **Shi et al. (2019)** — *Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition*, CVPR 2019
- **Gu et al. (2021)** — *Efficiently Modeling Long Sequences with Structured State Spaces (S4)*, ICLR 2022
- **Gu & Dao (2023)** — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752

### Biomechanics and Balance Assistance

- **Pai & Patton (1997)** — *Center of Mass Dynamics in Slips and Falls*, J. Biomechanics
- **Hof et al. (2005)** — *The Condition for Dynamic Stability (Margin of Stability)*, J. Biomechanics
- **Vallery et al. (2013)** — *Exoskeletons for Balance Assistance*
- **Krishnan et al. (2019)** — *Shared Control for Balance Assistance*

### Latent State & Classification

- **Thrun, Burgard & Fox (2005)** — *Probabilistic Robotics*, MIT Press

### Datasets (Primary Sources)

- **Kepski & Kwolek (2014)** — URFD: Embedded system for fall detection using body-worn accelerometer and depth sensor, IDAACS
- **Charfi et al. (2013)** — Le2i fall detection dataset, J. Electronic Imaging
- **Martínez-Villaseñor et al. (2019)** — UP-Fall Detection Dataset: A Multimodal Approach, Sensors
- **Tran et al. (2018)** — CMDFall: A Multi-Modal Multi-View Dataset for Human Fall Analysis, ICPR
- **Alam et al. (2024)** — GMDCSA-24: A Dataset for Human Fall Detection in Videos, Data in Brief
- **Eraso et al. (2022)** — CAUCAFall Dataset, Mendeley Data
- **Zhang, Conly & Athitsos (2014)** — EDF and OCCU: Evaluating Depth-Based Methods for Fall Detection under Occlusions, ISVC
- **Auvinet et al. (2010)** — Multiple Cameras Fall Dataset (MCFD), Technical Report 1350, Université de Montréal

---

## Contact

**Martin Vogel** — Master's Student, Assistive Robotics
Kyushu Institute of Technology × Polytech Nancy
vogel.martin-romain7334@mail.kyutech.jp
https://github.com/martinvgl
