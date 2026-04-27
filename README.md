# Connectivity-Constrained Multi-Robot Exploration with Adaptive Formation Control

**ROB 516: Advanced Multi-Robot Systems — Final Project, Winter 2026**
**Thirulok Sundar, University of Michigan – Ann Arbor**

## Overview

This project investigates the coverage–connectivity trade-off in multi-robot exploration across five experimental conditions on three structured map types (corridor, rooms, open). The key contribution is an information-gain-weighted soft connectivity penalty that replaces the standard hard per-step veto, recovering near-full coverage while reducing disconnection events by up to 66%.

## Repository Structure

```
├── final_project/          # Simulation code
│   ├── pipeline.py         # Core simulation pipeline (all 5 conditions)
│   ├── run_experiments.py  # Experiment runner (45 runs)
│   ├── connectivity.py     # Graph connectivity utilities (λ2, components)
│   ├── formation.py        # Formation and cohesion control
│   └── maps.py             # Procedural map generation (corridor/rooms/open)
│
├── results_v2/             # Canonical experiment results (w_p=10, α=0.35)
│   ├── experiment_summary.json
│   ├── coverage_curves.png
│   ├── disconnections.png
│   ├── steps_to_90.png
│   ├── lambda2_curves.png
│   └── snapshots_*.png     # Exploration snapshots per condition/map/seed
│
├── ieeeconf/               # Paper and slides (LaTeX)
│   ├── paper.tex           # IEEE conference paper source
│   ├── paper.pdf           # Compiled paper
│   ├── slides.tex          # Beamer presentation source
│   ├── slides.pdf          # Compiled slides (11 slides, 15 min)
│   └── ieeeconf.cls        # IEEE conference LaTeX class
│
├── map_preview.png         # Map type overview figure
└── ROB516_MP5.pdf          # Final submission PDF
```

## Five Experimental Conditions

| # | Condition | Description |
|---|-----------|-------------|
| 1 | Pure Frontier | Unconstrained frontier exploration (coverage baseline) |
| 2 | Frontier + Conn | Frontier + IG-weighted soft connectivity penalty |
| 3 | Rigid Formation | Group convoy to same frontier cluster (zero disconnections) |
| 4 | Adaptive Formation | Per-robot frontier + cohesion pull + soft penalty |
| 5 | Adversarial | Condition 4 with one robot actively disconnecting |

## Running Experiments

```bash
pip install numpy scipy matplotlib sympy
cd final_project
python run_experiments.py --output-dir ../results_v2 --conn-penalty-weight 10 --alpha 0.35
```

Key parameters:
- `--conn-penalty-weight` — soft penalty weight $w_p$ (default: 10)
- `--alpha` — cohesion blending parameter (default: 0.35)
- `--num-robots` — number of robots (default: 4)
- `--r-comm` — communication radius in cells (default: 10)

## Key Results

| Condition | Corridor Coverage | Corridor Disconnections |
|-----------|:-----------------:|:-----------------------:|
| 1. Pure Frontier | 94.1% | 483 |
| 2. Frontier+Conn | 93.0% | **164** (−66%) |
| 3. Rigid Formation | 50.6% | **0** |
| 4. Adaptive Formation | 71.2% | 381 |
| 5. Adversarial | 66.6% | 484 |

## Algorithmic Contributions

1. **IG-weighted soft penalty** — scores actions by frontier progress scaled by information gain minus connectivity cost; raises coverage from ≤32% (hard veto) to ≥93%
2. **Multi-pass connectivity filter** — resolves relay-chain ordering deadlocks in rigid formation
3. **Corrected cohesion radius** — $r_\text{eff} = 13$ cells allows relay-chain spread without centroid clustering
