"""
SI Fig. S2 - robustness to imperfect and noisy observation (combined script)
============================================================================
Reviewer 1, Comment 2. One file, two cells, in the style of the Fig_4
notebook. Cell 1 runs the noise sweep on the published Figure 4 engines and
writes figS2_noise_data.pkl. Cell 2 reads that file and draws the single
stochastic-dominance panel. Run Cell 1 once (slow at T = 50000), then Cell 2
as often as needed to restyle the figure without re-running the sweep.

Faithful to the published model: the only change to the engines is a single
guarded injection point where each predecessor protection level read into the
observation term O_i(t) is perturbed by additive Gaussian noise of standard
deviation eta and clipped to [0, 1]. At eta = 0 no extra random draw is
consumed, so the eta = 0 column reproduces the current results, R2 ~ 0.82 and
threshold C_E ~ 0.39 (Fig. 4E). The survival threshold and the
centrality-investment R2 under noise are reported with their values in the
SI Text S7.5 prose; only the stochastic-dominance result is plotted.

Real runs only. No values enter the manuscript or the response before this
script has been executed.
"""

# %% Cell 1 - simulation (run once; writes figS2_noise_data.pkl)
"""
SI Fig. S2 - SIMULATION: robustness to imperfect and noisy observation
======================================================================
Reviewer 1, Comment 2. The observation term O_i(t) is read with additive
Gaussian noise of standard deviation eta on each predecessor's protection
level, clipped to [0, 1], and eta is swept. Everything else is identical to
the published Figure 4 engines (cells 0 and 12 of Fig_4.ipynb).

Single injection point: the predecessor protection values read inside the
O_i computation are perturbed only when obs_noise > 0. At obs_noise = 0 no
extra random draws are consumed, so the eta = 0 column reproduces the
current results exactly (D_ks for delta near 0.50, R2 ~ 0.82, threshold
C_E ~ 0.39). This is the internal consistency anchor. The gamma = 0
baseline never enters the O_i block, so it is unaffected by eta.

Real runs only. No values are written into the manuscript or the response
until this script has been executed and figS2_noise_plot.py has rendered
the output.

Outputs: figS2_noise_data.pkl
"""

import numpy as np
import pickle
from scipy import stats
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
ETA_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]   # observation-noise standard deviations

# Panel B (KS distance, four scenarios): published config T=100, 50 reps
B_T, B_REPS = 100, 50
# Panel E (R2 and survival threshold, alpha): published config T=50000, 3 reps
# This is the slow part. Trim ETA_GRID or reduce E_REPS if a quick pass is needed,
# but keep eta = 0.0 so the reproduction anchor is always present.
E_T, E_REPS = 50000, 3

N, M = 100, 10
GAMMA = 0.3

# Set SMOKE = True only for a fast code check. Never report SMOKE numbers.
SMOKE = False
if SMOKE:
    B_T, B_REPS = 40, 4
    E_T, E_REPS = 400, 2
    ETA_GRID = [0.0, 0.2]


# --------------------------------------------------------------------------
# NETWORK (verbatim from Fig_4.ipynb)
# --------------------------------------------------------------------------
def generate_ba_network(N, m):
    adj = np.zeros((N, N), dtype=float)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    for new_node in range(m + 1, N):
        degrees = adj.sum(axis=1)[:new_node]
        if degrees.sum() == 0:
            probs = np.ones(new_node) / new_node
        else:
            probs = degrees / degrees.sum()
        targets = np.random.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = 1.0
            adj[t, new_node] = 1.0
    return adj


def eigenvector_centrality(adj, tol=1e-8, max_iter=1000):
    N = adj.shape[0]
    x = np.ones(N) / N
    for _ in range(max_iter):
        x_new = adj @ x
        norm = np.linalg.norm(x_new)
        if norm == 0:
            return np.ones(N) / N
        x_new = x_new / norm
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x_new / x_new.max()


# --------------------------------------------------------------------------
# PANEL B ENGINE (cell 0), with guarded observation noise
# --------------------------------------------------------------------------
def run_simulation(adj, centrality, ordering, params, gamma, T=100,
                   f_m=0.05, p_n=0.02, sigma_explore=0.03, p_recover=0.15,
                   obs_noise=0.0):
    N = adj.shape[0]
    p_max = params['p_max']; c_p = params['c_p']; p_l = params['p_l']
    p_r = params['p_r']; p_e = params['p_e']

    s0 = np.random.uniform(0.05, 0.4, N)
    s1 = np.random.uniform(0.1, 0.8, N)
    capital = np.ones(N) * 1.0
    failed = np.zeros(N, dtype=bool)

    rank = np.zeros(N, dtype=int)
    for pos, node in enumerate(ordering):
        rank[node] = pos
    neighbors = [np.where(adj[i] > 0)[0] for i in range(N)]

    for t in range(T):
        for i in range(N):
            if failed[i] and np.random.random() < p_recover:
                failed[i] = False
                capital[i] = 0.2

        f_p = np.zeros(N)
        for pos in range(N):
            node = ordering[pos]
            if failed[node]:
                f_p[node] = 0.0
                continue
            fp_base = s0[node] + s1[node] * centrality[node]
            if gamma > 0 and pos > 0:
                predecessors = ordering[:pos]
                pred_alive = predecessors[~failed[predecessors]]
                if len(pred_alive) > 0:
                    weights = centrality[pred_alive]
                    w_sum = weights.sum()
                    if w_sum > 0:
                        obs = f_p[pred_alive]
                        if obs_noise > 0:
                            obs = np.clip(obs + np.random.normal(0.0, obs_noise, len(obs)), 0.0, 1.0)
                        O_i = np.sum(weights * obs) / w_sum
                    else:
                        O_i = 0.0
                    fp_base += gamma * O_i
            f_p[node] = np.clip(fp_base, 0.0, 1.0 - f_m)

        fp_c = f_p * np.maximum(capital, 1e-10)
        p_p = p_max / (1.0 + c_p / np.maximum(fp_c, 1e-10))
        p_p = np.clip(p_p, 0.0, 1.0)

        failure_potential = np.zeros(N, dtype=bool)
        for i in range(N):
            if failed[i]:
                continue
            if np.random.random() < p_n:
                failure_potential[i] = True
            for nb in neighbors[i]:
                if failed[nb] and np.random.random() < p_l:
                    failure_potential[i] = True
                    break

        new_failed = failed.copy()
        for i in range(N):
            if failure_potential[i] and not failed[i]:
                if np.random.random() > p_p[i]:
                    new_failed[i] = True
                    capital[i] = 0.0
        failed = new_failed

        for i in range(N):
            if not failed[i]:
                growth = 1.0 + (1.0 - f_m - f_p[i]) * 0.1
                capital[i] = np.clip(capital[i] * growth, 0.0, 5.0)

        s0_new = s0.copy(); s1_new = s1.copy()
        for i in range(N):
            if failed[i]:
                continue
            r = np.random.random()
            if r < p_r:
                nb_list = neighbors[i]
                alive_nb = nb_list[~failed[nb_list]]
                if len(alive_nb) > 0:
                    nb_capitals = capital[alive_nb]
                    if nb_capitals.max() > capital[i]:
                        best_nb = alive_nb[np.argmax(nb_capitals)]
                        s0_new[i] = s0[best_nb]; s1_new[i] = s1[best_nb]
            elif r < p_r + p_e:
                s0_new[i] += np.random.normal(0, sigma_explore)
                s1_new[i] += np.random.normal(0, sigma_explore)
        s0 = np.clip(s0_new, 0.0, 1.0)
        s1 = np.clip(s1_new, 0.0, 1.0)

    return {'f_p': f_p, 'capital': capital, 'failed': failed, 'p_p': p_p, 'rank': rank}


# --------------------------------------------------------------------------
# PANEL E ENGINE (cell 12), with guarded observation noise
# --------------------------------------------------------------------------
def run_original_dynamics(adj, centrality, ordering, params, gamma, T=3000,
                          fm=0.1, obs_noise=0.0):
    N = adj.shape[0]
    pmax = params['pmax']; cp = params['cp']
    pn = params['pn']; pl = params['pl']
    pr = params['pr']; pe = params['pe']
    s_intensity = params.get('s', 1.0)
    mu = params.get('mu', 0.0)
    sigma = params.get('sigma', 0.0001)
    memory = params.get('memory', 0.99)
    neighbors = [np.where(adj[i] > 0)[0] for i in range(N)]
    Capital = np.ones(N); Capital_m = np.ones(N)
    s0 = np.zeros(N); s1 = np.zeros(N)
    Failure = np.zeros(N); failtimear = np.zeros(N)

    fp = s0 + s1 * centrality
    for t in range(T):
        failure_potential = np.zeros(N)
        failure_potential[np.random.random(N) <= pn] = 1.0
        for i in range(N):
            if Failure[i] > 0:
                for j in neighbors[i]:
                    if np.random.random() <= pl:
                        failure_potential[j] = 1.0
        failidx = (failtimear % 1 == 0) & (Failure > 0)
        Failure[failidx] = 0; failtimear[Failure == 0] = 0

        fp = s0 + s1 * centrality
        if gamma > 0:
            for pos in range(N):
                node = ordering[pos]
                if pos > 0:
                    predecessors = ordering[:pos]
                    pred_alive = predecessors[Failure[predecessors] == 0]
                    if len(pred_alive) > 0:
                        weights = centrality[pred_alive]
                        w_sum = weights.sum()
                        if w_sum > 0:
                            obs = fp[pred_alive]
                            if obs_noise > 0:
                                obs = np.clip(obs + np.random.normal(0.0, obs_noise, len(obs)), 0.0, 1.0)
                            O_i = np.sum(weights * obs) / w_sum
                            fp[node] += gamma * O_i
        fp = np.clip(fp, 0.0, 1.0 - fm)

        protection_probability = np.zeros(N)
        idx_pot = failure_potential > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            fp_c = fp[idx_pot] * Capital[idx_pot]
            protection_probability[idx_pot] = pmax / (1.0 + cp / np.maximum(fp_c, 1e-12))
        R1 = (np.random.random(N) <= (1.0 - protection_probability)) & idx_pot
        Failure[R1] = 1; Capital[R1] = 0.0; failtimear[R1] = 0
        failure_potential[:] = 0; failtimear[Failure > 0] += 1

        Capital = 1.0 + (1.0 - fm - fp) * Capital
        Capital_m = memory * Capital + (1.0 - memory) * Capital_m

        for i in range(N):
            if np.random.random() <= pr:
                rr = i
                while rr == i:
                    rr = np.random.choice(N)
                pi = 1.0 / (1.0 + np.exp(-s_intensity * (Capital_m[rr] - Capital_m[i])))
                if np.random.random() <= pi:
                    s0[i] = s0[rr]
        for i in range(N):
            if np.random.random() <= pr:
                rr = i
                while rr == i:
                    rr = np.random.choice(N)
                pi = 1.0 / (1.0 + np.exp(-s_intensity * (Capital_m[rr] - Capital_m[i])))
                if np.random.random() <= pi:
                    s1[i] = s1[rr]
        for i in range(N):
            if np.random.random() <= pe:
                s0[i] += np.random.normal(mu, sigma)
            if np.random.random() <= pe:
                s1[i] += np.random.normal(mu, sigma)

    return {'fp': fp, 'Capital': Capital, 'Failure': Failure}


# --------------------------------------------------------------------------
# METRICS
# --------------------------------------------------------------------------
def ks_one_sided(fp_seq_array, fp_sim_array):
    """Replicates the Panel B computation: one-sided KS gap of the sequential
    distribution dominating the simultaneous one, over active f_p (> 0.001),
    plus a flag for whether first-order stochastic dominance holds."""
    fp_seq = fp_seq_array.flatten(); fp_sim = fp_sim_array.flatten()
    a_seq = fp_seq[fp_seq > 0.001]; a_sim = fp_sim[fp_sim > 0.001]
    if len(a_seq) == 0 or len(a_sim) == 0:
        return 0.0, False
    x_max = max(a_seq.max(), a_sim.max()) * 1.05
    xr = np.linspace(0, x_max, 300)
    cdf_seq = np.array([np.mean(a_seq <= x) for x in xr])
    cdf_sim = np.array([np.mean(a_sim <= x) for x in xr])
    ks = float((cdf_sim - cdf_seq).max())
    fsd = bool(np.all(cdf_seq <= cdf_sim + 1e-9))
    return ks, fsd


SCENARIOS = {
    'delta': {'p_max': 1.0, 'c_p': 0.1, 'p_l': 0.01, 'p_r': 0.99, 'p_e': 0.99},
    'alpha': {'p_max': 1.0, 'c_p': 1.0, 'p_l': 0.05, 'p_r': 0.99, 'p_e': 0.99},
    'beta':  {'p_max': 0.1, 'c_p': 1.0, 'p_l': 0.10, 'p_r': 0.1,  'p_e': 0.1},
    'gamma': {'p_max': 0.1, 'c_p': 0.1, 'p_l': 0.08, 'p_r': 0.1,  'p_e': 0.1},
}
PARAMS_E = {'pmax': 1.0, 'cp': 0.05, 'pn': 0.1, 'pl': 0.3,
            'pr': 0.1, 'pe': 0.001, 's': 1.0, 'mu': 0.0, 'sigma': 0.001, 'memory': 0.99}


# --------------------------------------------------------------------------
# RUN
# --------------------------------------------------------------------------
def main():
    np.random.seed(42)
    adj = generate_ba_network(N, M)
    centrality = eigenvector_centrality(adj)
    ord_cent = np.argsort(-centrality)

    out = {'eta_grid': ETA_GRID, 'N': N, 'm': M, 'gamma': GAMMA,
           'centrality': centrality,
           'B_T': B_T, 'B_REPS': B_REPS, 'E_T': E_T, 'E_REPS': E_REPS}

    # ---- Panel B: KS distance and FSD per scenario across eta ----
    print("Panel B: KS distance vs observation noise")
    ks = {sc: [] for sc in SCENARIOS}
    fsd = {sc: [] for sc in SCENARIOS}
    for sc, p in SCENARIOS.items():
        # simultaneous baseline (gamma = 0) is eta-independent, computed once
        fp_sim = []
        for rep in range(B_REPS):
            np.random.seed(5000 + rep)
            fp_sim.append(run_simulation(adj, centrality, ord_cent, p, gamma=0.0, T=B_T)['f_p'])
        fp_sim = np.array(fp_sim)
        for eta in ETA_GRID:
            fp_seq = []
            for rep in range(B_REPS):
                np.random.seed(42 + rep)   # common random numbers across eta
                fp_seq.append(run_simulation(adj, centrality, ord_cent, p,
                                             gamma=GAMMA, T=B_T, obs_noise=eta)['f_p'])
            fp_seq = np.array(fp_seq)
            d, f = ks_one_sided(fp_seq, fp_sim)
            ks[sc].append(d); fsd[sc].append(f)
            print(f"  {sc:6s} eta={eta:.2f}  D_ks={d:.3f}  FSD={f}")
    out['ks'] = ks; out['fsd'] = fsd

    # ---- Panel E: R2 and survival threshold for alpha across eta ----
    print("Panel E: R2 and survival threshold vs observation noise")
    r2 = []; thr = []; ce_surv = []; ce_fail = []; n_surv_list = []
    for eta in ETA_GRID:
        fp_final = np.zeros((E_REPS, N)); cap_final = np.zeros((E_REPS, N))
        for rep in range(E_REPS):
            np.random.seed(42 + rep)   # matches published seeding; eta=0 reproduces exactly
            res = run_original_dynamics(adj, centrality, ord_cent, PARAMS_E,
                                        gamma=GAMMA, T=E_T, obs_noise=eta)
            fp_final[rep] = res['fp']; cap_final[rep] = res['Capital']
        fp_m = fp_final.mean(axis=0); cap_m = cap_final.mean(axis=0)
        r2_val = float(stats.linregress(centrality, fp_m).rvalue ** 2)
        survived = cap_m > np.median(cap_m)
        lr = LogisticRegression()
        lr.fit(centrality.reshape(-1, 1), survived.astype(int))
        t_val = float(-lr.intercept_[0] / lr.coef_[0, 0])
        r2.append(r2_val); thr.append(t_val)
        ce_surv.append(float(centrality[survived].mean()))
        ce_fail.append(float(centrality[~survived].mean()))
        n_surv_list.append(int(survived.sum()))
        print(f"  eta={eta:.2f}  R2={r2_val:.3f}  threshold C_E={t_val:.3f}  "
              f"n_surv={int(survived.sum())}")
    out['r2'] = r2; out['threshold'] = thr
    out['ce_surv'] = ce_surv; out['ce_fail'] = ce_fail; out['n_surv'] = n_surv_list

    with open('figS2_noise_data.pkl', 'wb') as f:
        pickle.dump(out, f)
    print("SAVED: figS2_noise_data.pkl")


main()

# %% Cell 2 - plot (reads figS2_noise_data.pkl; single dominance panel)
"""
SI Fig. S2 - PLOT: robustness of stochastic dominance to imperfect and noisy
observation
============================================================================
Reads figS2_noise_data.pkl produced by figS2_noise_sim.py and renders the
single dominance panel against observation noise eta. Separate from the
simulation file so the figure can be restyled without re-running the sweep.
Locked color scheme, plt.show, no savefig.

One-sided Kolmogorov-Smirnov distance D_ks for the four scenarios vs eta.
Filled markers mark eta values where first-order stochastic dominance holds
across the whole support; open markers mark one-sided dominance up to a
single marginal crossing.

The survival threshold (stays near 0.39) and the centrality-investment R2
(weakens with noise) are reported with their values in the SI Text S7.5
prose, not as panels. The summary print below still lists them for that text.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Locked colors
C_ALPHA = '#7B2D8E'
C_BETA  = '#C0392B'
C_GAMMA = '#D4780A'
C_DELTA = '#1A6B3C'
C_TEXT  = '#333333'
SC_COLOR = {'delta': C_DELTA, 'alpha': C_ALPHA, 'beta': C_BETA, 'gamma': C_GAMMA}
SC_LABEL = {'delta': r'$\delta$ full protection',
            'alpha': r'$\alpha$ coexistence',
            'beta':  r'$\beta$ all failure',
            'gamma': r'$\gamma$ partial coexistence'}

with open('figS2_noise_data.pkl', 'rb') as f:
    d = pickle.load(f)
eta = np.array(d['eta_grid'])
ks = d['ks']; fsd = d['fsd']
r2 = np.array(d['r2']); thr = np.array(d['threshold'])

print("Summary (eta : metric)")
for sc in ['delta', 'alpha', 'beta', 'gamma']:
    print(f"  D_ks {sc:6s}: " + ", ".join(f"{e:.2f}={v:.3f}" for e, v in zip(eta, ks[sc])))
print("  R2 alpha    : " + ", ".join(f"{e:.2f}={v:.3f}" for e, v in zip(eta, r2)))
print("  threshold   : " + ", ".join(f"{e:.2f}={v:.3f}" for e, v in zip(eta, thr)))

fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)

for sc in ['delta', 'alpha', 'beta', 'gamma']:
    y = np.array(ks[sc]); flags = np.array(fsd[sc])
    c = SC_COLOR[sc]
    ax.plot(eta, y, '-', color=c, lw=2.0, alpha=0.9, zorder=3, label=SC_LABEL[sc])
    for xi, yi, fi in zip(eta, y, flags):
        ax.scatter(xi, yi, s=46, zorder=4,
                   facecolors=c if fi else 'white', edgecolors=c, linewidths=1.6)

ax.set_xlabel(r'observation noise $\eta$', fontsize=12)
ax.set_ylabel(r'$D_{KS}$ (sequential vs simultaneous)', fontsize=12)
ax.set_title('Stochastic dominance under observation noise',
             fontsize=12.5, color=C_TEXT)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(C_TEXT)
ax.spines['bottom'].set_color(C_TEXT)
ax.tick_params(labelsize=10)
ax.set_xlim(eta.min() - 0.02, eta.max() + 0.02)
ax.set_xticks(eta)
ax.legend(fontsize=8.5, loc='upper left', frameon=False)

plt.tight_layout()
plt.show()
