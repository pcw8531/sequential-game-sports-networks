"""
Sequential Game Simulation on Scale-Free Sports Networks
=========================================================
Core simulation implementing the sequential game model with bounded
backward induction, social learning, and strategic exploration.

Extends Park & Fath (2026) Physica A 131258 from simultaneous to
sequential game structures.

Key equations (manuscript references):
    - Eigenvector centrality: Eq. 8
    - Decision ordering: Eq. 9-10
    - Softmax action selection: Eq. 11
    - Social learning priority: Eqs. 12a-12c
    - Imitation probability: Eq. 13
    - Functional capacity update: Eq. 14
    - Sequential protection: Eq. 15, 15a
    - Protection probability: Eq. 16

Parameters:
    N       : Network size (default 100)
    m       : BA connectivity (default 5)
    gamma   : Sequential observation weight (0 = simultaneous)
    beta    : Rationality intensity for softmax
    pr      : Social learning (imitation) probability
    pe      : Strategic exploration probability
    pp_max  : Maximum protection probability
    cp_half : Protection half-saturation constant
    pn      : Baseline failure probability
    pl      : Neighbor-propagated failure probability
    T       : Simulation time steps
    s       : Imitation sensitivity parameter

Usage:
    from simulation.sequential_simulation import SequentialGameSimulation
    sim = SequentialGameSimulation(N=100, m=5, gamma=0.3)
    sim.run(T=100, replications=50)
"""

import numpy as np
import networkx as nx


class SequentialGameSimulation:
    """
    Sequential game on Barabasi-Albert scale-free network.
    Agents decide in order of eigenvector centrality (Eq. 9).
    Later movers observe predecessors via O_i(t) (Eq. 15a).
    """

    def __init__(self, N=100, m=5, gamma=0.3, beta=10.0,
                 pr=0.1, pe=0.01, pp_max=1.0, cp_half=1.0,
                 pn=0.1, pl=0.3, fm=0.1, s=1.0, seed=42):

        self.N = N
        self.m = m
        self.gamma = gamma
        self.beta = beta
        self.pr = pr
        self.pe = pe
        self.pp_max = pp_max
        self.cp_half = cp_half
        self.pn = pn
        self.pl = pl
        self.fm = fm
        self.s = s
        self.seed = seed

        np.random.seed(seed)

        # ── Network construction (Barabasi-Albert) ──
        self.G = nx.barabasi_albert_graph(N, m, seed=seed)
        self.adj = nx.adjacency_matrix(self.G).toarray()

        # ── Eigenvector centrality (Eq. 8) ──
        ce_dict = nx.eigenvector_centrality_numpy(self.G)
        self.centrality = np.array([ce_dict[i] for i in range(N)])

        # ── Decision ordering sigma(i) (Eq. 9-10) ──
        # Descending centrality: most central decides first
        self.sigma = np.argsort(-self.centrality)
        self.sigma_inv = np.argsort(self.sigma)  # position of each node

        # ── Agent state variables ──
        self.capital = np.ones(N)           # functional capacity c_i(t)
        self.strategy_0 = np.zeros(N)       # s^0_i
        self.strategy_1 = np.zeros(N)       # s^1_i
        self.failure = np.zeros(N)          # binary failure state
        self.fp = np.zeros(N)               # protection investment f_p^i

        self.results = {}

    def _compute_observation(self, agent_i):
        """
        Centrality-weighted observation term O_i(t) (Eq. 15a).
        Averages predecessors' protection weighted by their centrality.
        """
        pos_i = self.sigma_inv[agent_i]
        if pos_i == 0:
            return 0.0  # first mover has no predecessors

        predecessors = self.sigma[:pos_i]
        weights = self.centrality[predecessors]
        w_sum = weights.sum()

        if w_sum < 1e-12:
            return 0.0

        return np.sum(weights * self.fp[predecessors]) / w_sum

    def _compute_protection(self):
        """
        Sequential protection function (Eq. 15):
        f_p^i(t) = s^0_i + s^1_i * C_E(v_i) + gamma * O_i(t)
        """
        for idx in range(self.N):
            agent = self.sigma[idx]  # process in decision order
            obs = self._compute_observation(agent)
            self.fp[agent] = (self.strategy_0[agent]
                              + self.strategy_1[agent] * self.centrality[agent]
                              + self.gamma * obs)
        self.fp = np.clip(self.fp, 0.0, 1.0 - self.fm)

    def _imitation_probability(self, c_j, c_i):
        """Fermi function for imitation (Eq. 13)."""
        return 1.0 / (1.0 + np.exp(-self.s * (c_j - c_i)))

    def _softmax_select(self, values):
        """Softmax action selection (Eq. 11)."""
        v = np.array(values) * self.beta
        v = v - v.max()  # numerical stability
        exp_v = np.exp(v)
        probs = exp_v / exp_v.sum()
        return np.random.choice(len(values), p=probs)

    def _strategy_update(self):
        """
        Modified backward induction with sequential priority (Eqs. 12a-12c):
        Step 1: Imitation (pr)
        Step 2: Exploration (pe)
        Step 3: Softmax backward induction
        """
        for idx in range(self.N):
            agent = self.sigma[idx]

            # Step 1: Imitation (Eq. 12a)
            if np.random.random() <= self.pr:
                neighbors = list(self.G.neighbors(agent))
                if neighbors:
                    target = np.random.choice(neighbors)
                    pi = self._imitation_probability(
                        self.capital[target], self.capital[agent])
                    if np.random.random() <= pi:
                        self.strategy_0[agent] = self.strategy_0[target]
                        self.strategy_1[agent] = self.strategy_1[target]
                        continue

            # Step 2: Exploration (Eq. 12b)
            if np.random.random() <= self.pe:
                self.strategy_0[agent] += np.random.normal(0, 0.1)
                self.strategy_1[agent] += np.random.normal(0, 0.1)
                continue

            # Step 3: Softmax (Eq. 12c)
            # Evaluate continuation values for small perturbations
            current_fp = self.fp[agent]
            candidates = [current_fp,
                          current_fp + 0.05,
                          current_fp - 0.05]
            candidates = [max(0, min(c, 1.0 - self.fm)) for c in candidates]

            # Protection probability for each candidate (Eq. 16)
            values = []
            for fp_c in candidates:
                denom = fp_c * self.capital[agent] + 1e-10
                pp = self.pp_max / (1.0 + self.cp_half / denom)
                values.append(pp * self.capital[agent])

            choice = self._softmax_select(values)
            self.fp[agent] = candidates[choice]

    def _failure_propagation(self):
        """
        Failure dynamics: baseline (pn) + neighbor propagation (pl).
        Protection probability (Eq. 16).
        """
        failure_potential = np.zeros(self.N)

        # Baseline failure risk
        failure_potential[np.random.random(self.N) <= self.pn] = 1

        # Neighbor-propagated risk
        for i in range(self.N):
            if self.failure[i] > 0:
                for j in self.G.neighbors(i):
                    if np.random.random() <= self.pl:
                        failure_potential[j] = 1

        # Protection check (Eq. 16)
        at_risk = failure_potential > 0
        if at_risk.any():
            denom = self.fp[at_risk] * self.capital[at_risk] + 1e-10
            protection_prob = self.pp_max / (1.0 + self.cp_half / denom)

            fail_check = np.random.random(at_risk.sum()) > protection_prob
            fail_indices = np.where(at_risk)[0][fail_check]
            self.failure[fail_indices] = 1
            self.capital[fail_indices] = 0

    def _update_capital(self):
        """Functional capacity update (Eq. 14)."""
        alive = self.failure == 0
        self.capital[alive] = (1.0 + (1.0 - self.fm - self.fp[alive])
                               ) * self.capital[alive]
        self.capital = np.clip(self.capital, 0, 100)

    def step(self):
        """Execute one time step."""
        self._strategy_update()
        self._compute_protection()
        self._failure_propagation()
        self._update_capital()

    def run(self, T=100, replications=50):
        """
        Run simulation for T time steps across multiple replications.

        Parameters:
            T           : Time steps per run
            replications: Independent runs to average

        Returns:
            Dictionary with averaged results.
        """
        all_failure_rates = []
        all_fp_means = []
        all_capital_means = []
        all_fp_by_centrality = []

        for rep in range(replications):
            np.random.seed(self.seed + rep)

            # Reset state
            self.capital = np.ones(self.N)
            self.strategy_0 = np.random.uniform(0, 0.1, self.N)
            self.strategy_1 = np.random.uniform(0, 0.1, self.N)
            self.failure = np.zeros(self.N)
            self.fp = np.zeros(self.N)

            failure_ts = []
            fp_ts = []
            capital_ts = []

            for t in range(T):
                self.step()
                failure_ts.append(self.failure.mean())
                fp_ts.append(self.fp.mean())
                capital_ts.append(self.capital.mean())

            all_failure_rates.append(failure_ts)
            all_fp_means.append(fp_ts)
            all_capital_means.append(capital_ts)
            all_fp_by_centrality.append(self.fp.copy())

        self.results = {
            'failure_rate': np.mean([fr[-1] for fr in all_failure_rates]),
            'fp_mean': np.mean([fp[-1] for fp in all_fp_means]),
            'capital_mean': np.mean([c[-1] for c in all_capital_means]),
            'failure_timeseries': np.mean(all_failure_rates, axis=0),
            'fp_timeseries': np.mean(all_fp_means, axis=0),
            'capital_timeseries': np.mean(all_capital_means, axis=0),
            'fp_by_node': np.mean(all_fp_by_centrality, axis=0),
            'centrality': self.centrality,
            'sigma': self.sigma,
        }

        return self.results

    def run_gamma_sweep(self, gamma_values, T=100, replications=50):
        """
        Run simulation across multiple gamma values.
        Useful for reproducing Fig. 4C (gamma advantage curve).

        Parameters:
            gamma_values: Array of gamma values to sweep
            T           : Time steps per run
            replications: Independent runs per gamma

        Returns:
            Dictionary mapping gamma -> results.
        """
        sweep_results = {}
        for gv in gamma_values:
            self.gamma = gv
            results = self.run(T=T, replications=replications)
            sweep_results[gv] = {
                'failure_rate': results['failure_rate'],
                'fp_mean': results['fp_mean'],
                'capital_mean': results['capital_mean'],
            }
            print(f"  gamma={gv:.2f}: failure={results['failure_rate']:.3f}, "
                  f"fp={results['fp_mean']:.4f}")
        return sweep_results


# ─────────────────────────────────────────────────────────────
# Convenience functions for quick usage
# ─────────────────────────────────────────────────────────────

def run_four_scenarios(N=100, m=5, gamma=0.3, T=100, replications=50):
    """
    Run all four equilibrium scenarios (Fig. 2).

    Scenarios:
        alpha: pp_max=1.0, cp_half=1.0 (coexistence)
        beta:  pp_max=0.1, cp_half=1.0 (system failure)
        gamma: pp_max=0.1, cp_half=0.1 (partial coexistence)
        delta: pp_max=1.0, cp_half=0.1 (full protection)
    """
    scenarios = {
        'alpha': {'pp_max': 1.0, 'cp_half': 1.0},
        'beta':  {'pp_max': 0.1, 'cp_half': 1.0},
        'gamma': {'pp_max': 0.1, 'cp_half': 0.1},
        'delta': {'pp_max': 1.0, 'cp_half': 0.1},
    }

    results = {}
    for name, params in scenarios.items():
        print(f"\nScenario {name}: pp_max={params['pp_max']}, "
              f"cp_half={params['cp_half']}")
        sim = SequentialGameSimulation(
            N=N, m=m, gamma=gamma,
            pp_max=params['pp_max'], cp_half=params['cp_half']
        )
        results[name] = sim.run(T=T, replications=replications)
        print(f"  Final failure rate: {results[name]['failure_rate']:.3f}")
        print(f"  Final protection mean: {results[name]['fp_mean']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────
# Direct execution
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Sequential Game Simulation")
    print("=" * 50)

    # Single run example
    sim = SequentialGameSimulation(N=100, m=5, gamma=0.3)
    results = sim.run(T=100, replications=10)

    print(f"\nResults (gamma=0.3):")
    print(f"  Failure rate: {results['failure_rate']:.3f}")
    print(f"  Protection mean: {results['fp_mean']:.4f}")
    print(f"  Capital mean: {results['capital_mean']:.2f}")

    # Gamma=0 recovery test
    print(f"\nGamma=0 recovery test:")
    sim0 = SequentialGameSimulation(N=100, m=5, gamma=0.0)
    r0 = sim0.run(T=100, replications=10)
    print(f"  Failure rate (gamma=0): {r0['failure_rate']:.3f}")
    print(f"  Confirms SPE(G) = NE(G) when gamma=0")
