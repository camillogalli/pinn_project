import numpy as np
import matplotlib.pyplot as plt
from src.physics.mmt_solver import power_law, flux_ab
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def power_law_derivatives_linear(k, p, max_order, k_min, k_max):

    domain = np.linspace(k_min, k_max, 1000)
    N = np.trapezoid(domain**p, domain)

    derivs = []
    def nk(k,p): return N*(k ** p)
    def dk_nk(k,p): return N*(k ** (p-1)) *p 
    def d2k_nk2(k,p): return N*p*(p-1)*(k**(p-2))
    def d3k_n_k3(k,p): return N*p*(p-1)*(p-2)*(k**(p-3))

    for order in range(max_order + 1):
        derivs.append([nk, dk_nk, d2k_nk2, d3k_n_k3][order](k, p))
    return derivs

def fit_operators_at_fixed_p_linear(p, A, k, k_min, k_max, beta=0.25,
                                N=300, IntN=200, max_order=3, ridge_alpha = 1e-3):
    
    k = np.asarray(k)

    derivs = power_law_derivatives_linear(k, p, max_order, k_min, k_max)

    F_true = np.array([
        flux_ab(power_law, (p, A), [0, kc], [kc, np.inf], beta, "Action", N=N, IntN=IntN)[0]
        for kc in k
    ])

    X = np.stack(derivs, axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg_exact = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(X_scaled, F_true)

    r2_exact = reg_exact.score(X_scaled, F_true)
    F_fit = reg_exact.predict(X_scaled)
    rel_error = np.linalg.norm(F_fit - F_true) / (np.linalg.norm(F_true) + 1e-20)

    return (reg_exact.coef_, reg_exact.intercept_, r2_exact, rel_error)

def plot_coefficients_vs_p_linear(ps, A, k, k_min, k_max, beta=0.25,
                               N=300, IntN=200, max_order=3, save_path=None, show=True):
    ps = np.asarray(ps)

    labels = [r"$n_k$", r"$\partial_k n_k$", r"$\partial^2_k n_k$", r"$\partial^3_k n_k$"
              ][:max_order + 1]

    res = { "exact": {lab: [] for lab in labels + ["int"] }}
    r2_exact_list = []
    rel_err_exact_list = []

    for p in ps:
        (coef_e, int_e, r2_e, rel_e) = fit_operators_at_fixed_p_linear(
            p, A, k, k_min, k_max, beta=beta, N=N, IntN=IntN, max_order=max_order, ridge_alpha = 1e-3
        )
        for lab, c in zip(labels, coef_e):
            res["exact"][lab].append(c)

        res["exact"]["int"].append(int_e)
        rel_err_exact_list.append(rel_e)
        r2_exact_list.append(r2_e)

    plt.figure(figsize=(14, 5))

    for lab in labels + ["int"] :
        plt.plot(ps, res["exact"][lab], "o-", markersize = 3, label = lab)

    plt.axhline(0, color="black", linewidth=0.6)
    plt.xlabel(r"$p$")
    plt.title("Flux prediction, linear model")
    plt.axvline(-5/6, linestyle="--", linewidth=0.8, color="black", label="Energy KZ Sol.")
    plt.axvline(-1/2, linestyle="-.", linewidth=0.8, color="black", label="Energy Equipartition")
    plt.axvline(-2/3, linestyle=":", linewidth=0.8, color="black", label="Action KZ Sol.")
    plt.xlim(-0.95, -0.4)
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
