import torch
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

def _build_operators_at_point(model, t, k_point):
    """Operatori differenziali di n_k (rete allenata) in un punto k_point, per un batch di t."""
    k = torch.full_like(t, k_point, requires_grad=True)
    inputs = torch.cat([t, k], dim=1)
    log_n = model(inputs)  # log(n_k) se la rete lavora in log-spazio
    n = torch.exp(log_n)

    dn_dk = torch.autograd.grad(n, k, grad_outputs=torch.ones_like(n), create_graph=True)[0]
    d2n_dk2 = torch.autograd.grad(dn_dk, k, grad_outputs=torch.ones_like(dn_dk), create_graph=True)[0]
    d3n_dk3 = torch.autograd.grad(d2n_dk2, k, grad_outputs=torch.ones_like(d2n_dk2), create_graph=True)[0]

    return {"n": n, "dn_dk": dn_dk, "d2n_dk2": d2n_dk2, "d3n_dk3": d3n_dk3, "n_dn_dk": n * dn_dk}


def build_operator_library_two_points(model, t, k_max, k_min):
    """Libreria operatori dalla rete, valutata a k_max e k_min (per uso su MMT/rete vera)."""
    lib_kmax = _build_operators_at_point(model, t, k_max)
    lib_kmin = _build_operators_at_point(model, t, k_min)
    library = {f"{name}_kmax": val for name, val in lib_kmax.items()}
    library.update({f"{name}_kmin": val for name, val in lib_kmin.items()})
    return library


def analytical_operator_library(alphas, A, k_max, k_min):
    """
    Libreria operatori calcolata ANALITICAMENTE per n_k = A * k^-alpha,
    campionando alphas indipendenti (non un'unica traiettoria temporale).
    Usata per validare, sul caso del trasporto, la pipeline di model
    comparison prima di applicarla al dataset MMT.
    """
    alphas = torch.as_tensor(alphas, dtype=torch.float32).reshape(-1, 1)

    def n(k, a):   return A * k ** (-a)
    def dn(k, a):  return -a * A * k ** (-a - 1)
    def d2n(k, a): return a * (a + 1) * A * k ** (-a - 2)
    def d3n(k, a): return -a * (a + 1) * (a + 2) * A * k ** (-a - 3)

    library = {
        "n_kmax": n(k_max, alphas), "n_kmin": n(k_min, alphas),"dn_dk_kmax": dn(k_max, alphas),"dn_dk_kmin": dn(k_min, alphas),
        "d2n_dk2_kmax": d2n(k_max, alphas),"d2n_dk2_kmin": d2n(k_min, alphas), "d3n_dk3_kmax": d3n(k_max, alphas),
        "d3n_dk3_kmin": d3n(k_min, alphas)
    }
    F_true = n(k_max, alphas) - n(k_min, alphas)
    return library, F_true

def fit_operators_at_fixed_alpha(model, sensor_grid, alpha, A, k_c_values, k_min, include_d2=True):
    """
    Per un SINGOLO alpha fissato, fitta F(k_c) (predetto dal DeepONet) come
    combinazione lineare di n_k(k_c), dn_k/dk(k_c), [d2n_k/dk2(k_c)], con intercetta.
    """
    k_c_values = np.asarray(k_c_values)

    # operatori analitici, valutati in funzione di k_c, per questo singolo alpha
    n_vals = A * k_c_values ** (-alpha)
    dn_vals = -alpha * A * k_c_values ** (-alpha - 1)
    d2n_vals = alpha * (alpha + 1) * A * k_c_values ** (-alpha - 2)

    # profilo n_k(.) per QUESTO alpha, campionato sui sensori — un solo profilo,
    # ripetuto una volta per ciascun k_c_value (input del branch net è costante)
    profile = A * sensor_grid ** (-alpha)
    profile_t = torch.tensor(profile, dtype=torch.float32).unsqueeze(0).repeat(len(k_c_values), 1)
    k_c_t = torch.tensor(k_c_values, dtype=torch.float32).reshape(-1, 1)

    with torch.no_grad():
        F_pred = model(profile_t, k_c_t).squeeze().numpy()

    F_exact = A * (k_c_values ** (-alpha) - k_min ** (-alpha))

    features = [n_vals] #dn_vals]
    #if include_d2:
    #    features.append(d2n_vals)
    X = np.stack(features, axis=1)

    reg_pred = LinearRegression(fit_intercept=True).fit(X, F_pred)
    reg_exact = LinearRegression(fit_intercept=True).fit(X, F_exact)

    return (reg_pred.coef_, reg_pred.intercept_), (reg_exact.coef_, reg_exact.intercept_)

def plot_coefficients_vs_alpha(model, sensor_grid, alphas, A, k_c_values, k_min, save_path=None, show=True):
    alphas = np.asarray(alphas)

    res = {"pred": {"a": [], "b": [], "c": [], "int": []},
           "exact": {"a": [], "b": [], "c": [], "int": []}}

    for alpha in alphas:
        (coef_p, int_p), (coef_e, int_e) = fit_operators_at_fixed_alpha(
            model, sensor_grid, alpha, A, k_c_values, k_min
        )
        for key, coef, inter in (("pred", coef_p, int_p), ("exact", coef_e, int_e)):
            res[key]["a"].append(coef[0])
            #res[key]["b"].append(coef[1])
            #res[key]["c"].append(coef[2])
            res[key]["int"].append(inter)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, key, titolo in (
        (ax1, "pred", "Fit su $F$ predetto (DeepONet)"),
        (ax2, "exact", "Fit su $F$ analitico"),
    ):
        ax.plot(alphas, res[key]["a"], "o-", markersize=3, label=r"$a(\alpha)$ — $n_k$")
        #ax.plot(alphas, res[key]["b"], "o-", markersize=3, label=r"$b(\alpha)$ — $\partial_k n_k$")
        #ax.plot(alphas, res[key]["c"], "o-", markersize=3, label=r"$c(\alpha)$ — $\partial_k^2 n_k$")
        ax.plot(alphas, res[key]["int"], "o-", markersize=3, label=r"intercetta$(\alpha)$")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axhline(1, color="gray", linewidth=0.6, linestyle=":")
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(titolo)

    ax1.set_ylabel("coefficiente")
    ax1.set_ylim(bottom=-8, top=8)
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()