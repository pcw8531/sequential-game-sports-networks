# Sequential Game Dynamics Produce Subgame Perfect Equilibria in Sports Networks

**Code and data repository for:** "Sequential Game Dynamics Produce Subgame Perfect Equilibria in Sports Networks Through Bounded Rationality"

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19528357.svg)](https://doi.org/10.5281/zenodo.19528357)

**Authors:** Chulwook Park

## Overview

This repository provides simulation code, figure generation scripts, and supplementary materials for a study extending bounded rationality game theory from simultaneous to sequential structures on scale-free sports networks. The model shows how modified backward induction yields ε-subgame perfect equilibria that converge to exact subgame perfect equilibria (SPE) as the rationality parameter increases, and these constitute refined subsets of Nash equilibria (NE), with SPE(Γ) ⊆ NE(Γ).

**Key contributions:**
- Centrality-based decision ordering σ(i) on Barabási–Albert networks yields ε-SPE that converge to exact SPE through bounded backward induction
- Sequential play shows first-order stochastic dominance over the simultaneous baseline in all four equilibrium scenarios examined
- The centrality–protection relationship (R² = 0.82) appears across the banking, epidemiology, and sports domains examined
- Setting γ = 0 exactly recovers the published simultaneous model, confirming theoretical continuity
- Robustness to imperfect observation: the stochastic dominance and the survival threshold persist under observation noise, while the graded centrality–investment correlation is sensitive to it (SI Fig. S2)

**Foundation:** This work extends [Park & Fath (2026), Physica A, 131258](https://github.com/pcw8531/nash-equilibria-sports-networks) from simultaneous to sequential game structures.

## Repository Structure

```
sequential-game-sports-networks/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── simulation/
│   ├── 02_sequential_simulation.py    # Full sequential model with backward induction
│   └── figS2_noise_robustness.py      # Observation-noise robustness analysis (SI Fig. S2)
│
├── supplement/
│   ├── SI_Movie_S1.gif               # Animation: attractor evolution (Fig 3D)
│   ├── SI_Movie_S2.gif               # Animation: 4-scenario migration (Fig 4D)
│   ├── gen_movie_s1.py               # Movie S1 generation code
│   └── gen_movie_s2.py               # Movie S2 generation code
│
└── data/
    └── README.md                      # Data provenance and source documentation
```

## Installation

```bash
git clone https://github.com/pcw8531/sequential-game-sports-networks.git
cd sequential-game-sports-networks
pip install -r requirements.txt
```

## Usage

The core simulation model is implemented in `simulation/02_sequential_simulation.py`. Figure generation scripts are maintained locally and available upon request. The core simulation code in `simulation/` reproduces all computational results.

### Robustness to observation noise
`simulation/figS2_noise_robustness.py` is a single two-cell script. The first cell runs the noise sweep on the published Figure 4 engines and writes `figS2_noise_data.pkl`, and the second cell reads that file and draws the stochastic-dominance panel reported as SI Fig. S2. The survival threshold and the centrality–investment R² under noise are printed to the console and reported in SI Text S7.5.
```bash
python simulation/figS2_noise_robustness.py   # SI Fig. S2: robustness to observation noise
```

### Generate supplementary movies
```bash
python supplement/gen_movie_s1.py     # SI Movie S1: attractor evolution
python supplement/gen_movie_s2.py     # SI Movie S2: 4-scenario migration
```

## Model Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Network size | N | 100 | Agents in scale-free network |
| Connectivity | m | 10 | Edges per new node (BA model) |
| Max protection | p_p,max | 0.1–1.0 | Protection capacity ceiling |
| Protection scaling | c_p,1/2 | 0.1–1.0 | Half-saturation constant |
| Imitation rate | p_r | 0.01–0.99 | Social learning intensity |
| Exploration rate | p_e | 0.01–0.99 | Strategic innovation rate |
| Rationality | β | 0.1–50.0 | Backward induction precision |
| Observation weight | γ | 0.0–1.0 | Sequential sensitivity (γ=0 → simultaneous) |

## Four Equilibrium Scenarios

| Scenario | Color | p_p,max | c_p,1/2 | Outcome |
|----------|-------|---------|---------|---------|
| α Coexistence | #7B2D8E | 1.0 | 1.0 | Mixed protection-failure equilibrium |
| β System failure | #C0392B | 0.1 | 1.0 | Network-wide cascading failure |
| γ Partial coexistence | #D4780A | 0.1 | 0.1 | Limited partial recovery |
| δ Full protection | #1A6B3C | 1.0 | 0.1 | Complete network protection |

## Supplementary Movies

- **SI Movie S1:** Continuous attractor evolution across p_l = 0.1 → 0.5, showing flow field redirection, network failure propagation from hub-protected to failure-dominated topology under increasing contagion risk (animated version of Figure 3A)
- **SI Movie S2:** Simultaneous four-scenario (δ, α, γ, β) attractor migration from γ = 0 to 1.0, with scenario-specific trajectories, embedded networks, and dynamic flow fields (animated version of Figure 4D)

## Related Repository

- **Simultaneous model (Physica A):** [nash-equilibria-sports-networks](https://github.com/pcw8531/nash-equilibria-sports-networks) — the foundation from which this sequential extension is built. Setting γ = 0 in this repository exactly reproduces the simultaneous model results.

## Citation

If you use this code, please cite:

```
Park, C. (2026). Sequential game dynamics produce subgame perfect equilibria in sports
networks through bounded rationality. [Manuscript submitted for publication].
```

And the foundation paper:

```
Park, C., & Fath, B. D. (2026). Bounded rationality produces Nash equilibria
in sports networks: Protection, learning, and strategic adaptation.
Physica A, 131258.
```

## License

MIT License — see [LICENSE](LICENSE) for details.
