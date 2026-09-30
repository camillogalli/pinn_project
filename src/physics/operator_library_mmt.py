import torch
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from src.physics.mmt_solver import power_law, flux_ab
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def power_law_derivatives(k, p, A, max_order):

    derivs = []
    def n3k(k,p): return (A * k ** p)**3
    def n2k_dk_nk(k,p): return (A * k ** p)**2 * (p * A * k ** (p - 1))
    def nk_dk_nk2(k,p): return (A * k ** p) * (p * A * k ** (p - 1))**2
    def dk_nk3(k,p): return (p * A * k ** (p - 1))**3

    for order in range(max_order + 1):
        derivs.append([n3k, n2k_dk_nk, nk_dk_nk2, dk_nk3][order](k, p))
    return derivs  


def fit_operators_at_fixed_p(model, sensor_grid, p, A, k_c_values, beta=0.25,
                                N=300, IntN=200, max_order=3, ridge_alpha = 1e-3):
    
    k_c_values = np.asarray(k_c_values)

    derivs = power_law_derivatives(k_c_values, p, A, max_order)

    profile = power_law(p, A, sensor_grid)
    profile_t = torch.tensor(profile, dtype=torch.float32).unsqueeze(0).repeat(len(k_c_values), 1)
    k_c_t = torch.tensor(k_c_values, dtype=torch.float32).reshape(-1, 1)

    with torch.no_grad():
        F_pred = model(profile_t, k_c_t).squeeze().numpy()

    F_true = np.array([
        flux_ab(power_law, (p, A), [0, kc], [kc, np.inf], beta, "Action", N=N, IntN=IntN)[0]
        for kc in k_c_values
    ])

    X = np.stack(derivs, axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg_pred = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(X_scaled, F_pred)
    reg_exact = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(X_scaled, F_true)

    r2_pred = reg_pred.score(X_scaled, F_pred)
    r2_exact = reg_exact.score(X_scaled, F_true)

    return (reg_pred.coef_, reg_pred.intercept_, r2_pred), (reg_exact.coef_, reg_exact.intercept_, r2_exact)

def plot_flux_reconstructed(model, sensor_grid, k_c_values, p= -5/6, A = 1, beta = 0.25, N = 300, IntN = 200,
                             max_order = 3, save_path=None, show=True):

    k_c_values = np.asarray(k_c_values)

    (coef_p, int_p, r2_p), (coef_e, int_e, r2_e) = fit_operators_at_fixed_p(
            model, sensor_grid, p, A, k_c_values, beta=beta, N=N, IntN=IntN, max_order=max_order, ridge_alpha = 1e-3
    )

    derivs = power_law_derivatives(k_c_values, p, A, max_order)
    X = np.stack(derivs, axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    F_reconstructed = X_scaled @ coef_e + int_e
    F1 = flux_ab(power_law, (p, A), [0, 1], [1, np.inf], beta, "Action", N=N, IntN=IntN)[0]
    F_real = F1*k_c_values ** (-1/2)
 
    plt.figure(figsize=(7,5))
    plt.plot(k_c_values, F_reconstructed , linewidth = 3, label = "Reconstructed flux", color = "steelblue")
    plt.plot(k_c_values, F_real, linewidth = 3, linestyle ="--", label = "True flux", color = "orange")
    plt.xlabel(r"$k_c$")
    plt.ylabel(r"$F$")
    plt.title(r"Reconstructed flux vs $k_c$ at $p_{kz}=-5/6$")
    plt.grid(color="lightgrey", linestyle="-", linewidth=0.4)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()

def plot_coefficients_vs_p(model, sensor_grid, ps, A, k_c_values, beta=0.25,
                               N=300, IntN=200, max_order=3, save_path=None, show=True):
    ps = np.asarray(ps)

    labels = [r"$(n_k)^3$", r"$n_k^2(\partial_k n_k)$", r"$(\partial_k n_k)^2 n_k$",
              r"$(\partial_k n_k)^3$"][:max_order + 1]

    res = {"pred": {lab: [] for lab in labels + ["int"]},
           "exact": {lab: [] for lab in labels + ["int"]}}
    r2_pred_list, r2_exact_list = [], []

    for p in ps:
        (coef_p, int_p, r2_p), (coef_e, int_e, r2_e) = fit_operators_at_fixed_p(
            model, sensor_grid, p, A, k_c_values, beta=beta, N=N, IntN=IntN, max_order=max_order, ridge_alpha = 1e-3
        )
        for key, coef, inter in (("pred", coef_p, int_p), ("exact", coef_e, int_e)):
            for lab, c in zip(labels, coef):
                res[key][lab].append(c)
            res[key]["int"].append(inter)
        r2_pred_list.append(r2_p)
        r2_exact_list.append(r2_e)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, key, titolo in ((ax1, "pred", "Fit su $F$ predetto (DeepONet)"),
                             (ax2, "exact", "Fit su $F$ solver MMT")):
        for lab in labels + ["int"]:
            ax.plot(ps, res[key][lab], "o-", markersize=3, label=lab)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel(r"$p$")
        ax.set_title(titolo)
        ax.set_ylim(-100, 100)
        ax.axvline(-5/6, linestyle="--", linewidth=0.8, color="black", label="Energy KZ Sol.")
        ax.axvline(-1/2, linestyle="-.", linewidth=0.8, color="black", label="Energy Equipartition")

    ax1.set_ylabel("coefficiente")
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def linear_reconstruction(k,p, k_min, k_max, a , b, c):

    def n_k(k,p) : return k**p
    def d_kn_k(k,p) : return p*(k**(p-1))
    def d2_kn_k(k,p): return p*(p-1)*(k**(p-2))

    domain = np.linspace(k_min, k_max, 10000)
    N = np.trapz(n_k(k, p), domain)  

    return a*N*n_k(k,p) + b*N*d_kn_k(k,p) + c*N*d2_kn_k(k,p)

