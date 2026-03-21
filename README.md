# Sequential Game Dynamics Produce Subgame Perfect Equilibria in Sports Networks

**Code and data repository for:** "Sequential Game Dynamics Produce Subgame Perfect Equilibria in Sports Networks Through Bounded Rationality"

**Authors:** Chulwook Park

## Overview

This repository provides simulation code, figure generation scripts, and supplementary materials for a study extending bounded rationality game theory from simultaneous to sequential structures on scale-free sports networks. The model demonstrates how subgame perfect equilibria (SPE) emerge through modified backward induction and constitute refined subsets of Nash equilibria (NE), establishing that SPE(Γ) ⊆ NE(Γ).

**Key contributions:**
- Centrality-based decision ordering σ(i) on Barabási–Albert networks produces SPE through bounded backward induction
- Sequential play generates first-order stochastic dominance over the simultaneous baseline across all four equilibrium scenarios
- The centrality–protection relationship (R² = 0.82) generalizes across banking, epidemiology, and sports domains
- Setting γ = 0 exactly recovers the published simultaneous model, confirming theoretical continuity

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
│   └── 02_sequential_simulation.py    # Full sequential model with backward induction
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

## Quick Start

### Run the sequential simulation
```python
from simulation.sequential_simulation import SequentialGameSimulation

# Initialize with default parameters (N=100, m=5, γ=0.3)
sim = SequentialGameSimulation(N=100, m=5, gamma=0.3)
sim.run(T=100, replications=50)

# Access results
print(f"SPE protection mean: {sim.results['fp_mean']:.4f}")
print(f"Failure rate: {sim.results['failure_rate']:.4f}")
```

### Reproduce figures
Each figure panel has a standalone script in `figures/`:
```bash
# Example: generate Figure 3 Panel D (ternary flow fields)
python figures/fig3/fig3_D_ternary.py
```

With:

### Generate supplementary movies
```bash
python supplement/gen_movie_s1.py     # SI Movie S1: attractor evolution
python supplement/gen_movie_s2.py     # SI Movie S2: 4-scenario migration
```

## Model Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Network size | N | 100 | Agents in scale-free network |
| Connectivity | m | 5 | Edges per new node (BA model) |
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
| γ Partial coexistence | #D4780A | 0.1 | 0.1 | Hub-vulnerable partial recovery |
| δ Full protection | #1A6B3C | 1.0 | 0.1 | Complete network protection |

## Supplementary Movies

- **SI Movie S1:** Continuous attractor evolution across γ = 0.01 → 0.9, showing flow field redirection, network failure propagation, and SPE = NE → SPE ⊂ NE transition (animated version of Figure 3D)
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

## Contact

- **Chulwook Park** — Seoul National University (BK21 Four) / OIST — pcw8531@snu.ac.kr
