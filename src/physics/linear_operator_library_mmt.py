import numpy as np
import matplotlib.pyplot as plt
from src.physics.mmt_solver import power_law, flux_ab
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

def power_law_derivatives_linear(k, p, max_order, k_min, k_max):

    domain = np.linspace(k_min, k_max, 3000)
    N_p = np.trapezoid(domain**p, domain)

    derivs = []
    def intercept(k,p): return np.full_like(k, N_p**3, dtype = float)
    def nk(k,p): return (N_p**2)*(k ** p)
    def dk_nk(k,p): return (N_p**2)*(k ** (p-1)) *p 
    def d2k_nk2(k,p): return (N_p**2)*p*(p-1)*(k**(p-2))
    def d3k_n_k3(k,p): return (N_p**2)*p*(p-1)*(p-2)*(k**(p-3))

    for order in range(max_order + 1):
        derivs.append([intercept, nk, dk_nk, d2k_nk2, d3k_n_k3][order](k, p))
    return derivs, N_p

def fit_operators_at_fixed_p_linear(p, A, k, k_min, k_max, beta=0.25,
                                N=300, IntN=200, max_order=3, ridge_alpha = 1e-3):
    
    k = np.asarray(k)

    derivs, N_p = power_law_derivatives_linear(k, p, max_order, k_min, k_max)

    F_true = np.array([
        flux_ab(power_law, (p, A), [0, kc], [kc, np.inf], beta, "Energy", N=N, IntN=IntN)[0]
        for kc in k
    ])

    X = np.stack(derivs, axis=1)
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)

    reg_exact = LinearRegression(fit_intercept=False).fit(X, F_true)

    r2_exact = reg_exact.score(X, F_true)
    F_fit = reg_exact.predict(X)
    rel_error = np.linalg.norm(F_fit - F_true) / (np.linalg.norm(F_true) + 1e-20)

    return (reg_exact.coef_, r2_exact, rel_error, N_p)

def plot_coefficients_vs_p_linear(ps, A, k, k_min, k_max, beta=0.25,
                               N=300, IntN=200, max_order=3, save_path=None, show=True):
    ps = np.asarray(ps)

    labels = ["int", r"$n_k$", r"$\partial_k n_k$", r"$\partial^2_k n_k$", r"$\partial^3_k n_k$"
              ][:max_order + 1]

    res = { "exact": {lab: [] for lab in labels}}
    N_p_list = []
    r2_exact_list = []
    rel_err_exact_list = []

    for p in ps:
        (coef_e, r2_e, rel_e, N_p) = fit_operators_at_fixed_p_linear(
            p, A, k, k_min, k_max, beta=beta, N=N, IntN=IntN, max_order=max_order, ridge_alpha = 1e-3
        )

        N_p_list.append(N_p)

        for lab, c in zip(labels, coef_e):
            res["exact"][lab].append(c)

        rel_err_exact_list.append(rel_e)
        r2_exact_list.append(r2_e)

    N_p_list = np.asarray(N_p_list)

    plt.figure(figsize=(14, 5))

    for lab in labels :
        plt.plot(ps, res["exact"][lab], "o-", markersize = 3, label = lab)

    plt.axhline(0, color="black", linewidth=0.6)
    plt.xlabel(r"$p$")
    plt.ylabel(r"coefficients")
    plt.title("MMT Flux reconstruction, linear model")
    plt.axvline(-5/6, linestyle="--", linewidth=0.8, color="black", label="Energy KZ Sol.")
    plt.axvline(-1/2, linestyle="-.", linewidth=0.8, color="black", label="Energy Equipartition")
    plt.axvline(-2/3, linestyle=":", linewidth=0.8, color="black", label="Action KZ Sol.")
    plt.xlim(-0.9, -0.3)
    plt.ylim(-10, 10)
    plt.legend(loc = "center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def fit_flux_through_1(ps, A, max_order = 4, beta=0.25,
                        N=300, IntN=200, save_path = None, show=True):

    ps = np.asarray(ps)

    F_true = np.array([
        flux_ab(power_law, (p, A), [0, 1], [1, np.inf], beta, "Action", N=N, IntN=IntN)[0]
        for p in ps
    ])

    def basis_row(p, L):
        row, term = [1.0], 1.0
        for ell in range(1, L + 1):
            term *= (p - (ell - 1))
            row.append(term)
        return row

    X = np.array([basis_row(p, max_order) for p in ps])


    reg_exact = LinearRegression(fit_intercept=False).fit(X, F_true)
    F_fit = reg_exact.predict(X)
    rel_error = np.linalg.norm(F_fit - F_true) / (np.linalg.norm(F_true) + 1e-30)

    plt.figure(figsize=(14,5))
    plt.title("MMT Flux reconstruction, linear model")
    plt.xlabel(r"$p$")
    plt.ylabel(r"Flux $\hat{F}(p)$")
    plt.axhline(0, color="black", linewidth=0.6)
    plt.plot(ps, F_true,  linewidth=3, label="True Flux")
    plt.plot(ps, F_fit,  linewidth=3, label="Fitted Flux")
    plt.axvline(-5/6, linestyle="--", linewidth=0.8, color="black", label="Energy KZ Sol.")
    plt.axvline(-1/2, linestyle="-.", linewidth=0.8, color="black", label="Energy Equipartition")
    plt.axvline(-2/3, linestyle=":", linewidth=0.8, color="black", label="Action KZ Sol.")
    plt.xlim(-0.9, 0.05)
    plt.ylim(-6, 15)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return reg_exact.coef_, rel_error

def fit_DAM(ps, k, A, beta, max_order, k_min, k_max, N=300, IntN=200, save_path=None, show=True):

    ps = np.asarray(ps)
    plt.figure(figsize=(14,6))

    coef_list = []
    err_list = []
    F_fit_list = []
    F_true_list = []

    for p in ps:

        F_true = flux_ab(power_law, (p, A), [0, 1.0], [1.0, np.inf], beta, "Energy", N=N, IntN=IntN)[0]
        F_true_list.append(F_true)

        (coef_e, r2_e, rel_e, N_p) = fit_operators_at_fixed_p_linear(
                p, A, k, k_min, k_max, beta=beta, N=N, IntN=IntN, max_order=max_order, ridge_alpha = 1e-3
            )

        derivs, N_p = power_law_derivatives_linear(1.0, p, max_order, k_min, k_max)
        derivs = np.asarray(derivs)[:len(coef_e)]

        F_fit_p = np.dot(coef_e, derivs)
        F_fit_list.append(F_fit_p)

        coef_list.append(coef_e)
        err_list.append(rel_e)

    F_true = np.asarray(F_true_list)
    F_fit = np.asarray(F_fit_list)

    global_error = (np.linalg.norm(F_fit - F_true)/ (np.linalg.norm(F_true) + 1e-30))


    plt.plot(ps, F_fit, linestyle="--", linewidth=2, label=f"Reconstructed flux, error={global_error:.3e}")
    plt.plot(ps, F_true,  linewidth=3, label="True Flux")
    plt.xlabel(r"$p$")
    plt.ylabel(r"Flux $\hat{F}(p)$")
    plt.title("Reconstructed MMT Flux through linear differential model, max order = {}".format(max_order))
    plt.axhline(0, color="black", linewidth=0.6)   
    plt.axvline(-5/6, linestyle="--", linewidth=0.8, color="black", label="Energy KZ Sol.")
    plt.axvline(-1/2, linestyle="-.", linewidth=0.8, color="black", label="Energy Equipartition")
    plt.axvline(-2/3, linestyle=":", linewidth=0.8, color="black", label="Action KZ Sol.")
    plt.xlim(-0.9, 0.05)
    plt.ylim(-6, 15)
    plt.grid(color="lightgrey", linestyle="-", linewidth=0.4)
    plt.legend()
    plt.tight_layout()   
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
