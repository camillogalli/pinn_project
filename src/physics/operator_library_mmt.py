import torch
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from src.physics.mmt_solver import power_law, flux_ab

def power_law_derivatives(k, p, A, max_order):

    derivs = [A * k ** (p)]
    coef = 1.0
    for order in range(1, max_order + 1):
        coef *= (p - (order - 1))
        derivs.append(coef * A * k ** (p - order))
    return derivs  

def fit_operators_at_fixed_p(model, sensor_grid, p, A, k_c_values, beta=0.25,
                                N=300, IntN=200, max_order=4):
    
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

    reg_pred = LinearRegression(fit_intercept=True).fit(X, F_pred)
    reg_exact = LinearRegression(fit_intercept=True).fit(X, F_true)

    r2_pred = reg_pred.score(X, F_pred)
    r2_exact = reg_exact.score(X, F_true)

    return (reg_pred.coef_, reg_pred.intercept_, r2_pred), (reg_exact.coef_, reg_exact.intercept_, r2_exact)
    

def plot_coefficients_vs_alpha(model, sensor_grid, ps, A, k_c_values, beta=0.25,
                               N=300, IntN=200, max_order=4, save_path=None, show=True):
    ps = np.asarray(ps)

    labels = [r"$n_k$", r"$\partial_k n_k$", r"$\partial_k^2 n_k$",
              r"$\partial_k^3 n_k$", r"$\partial_k^4 n_k$"][:max_order + 1]

    res = {"pred": {lab: [] for lab in labels + ["int"]},
           "exact": {lab: [] for lab in labels + ["int"]}}
    r2_pred_list, r2_exact_list = [], []

    for p in ps:
        (coef_p, int_p, r2_p), (coef_e, int_e, r2_e) = fit_operators_at_fixed_p(
            model, sensor_grid, p, A, k_c_values, beta=beta, N=N, IntN=IntN, max_order=max_order
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

    ax1.set_ylabel("coefficiente")
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()