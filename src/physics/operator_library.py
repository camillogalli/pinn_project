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

def fit_operators_at_fixed_alpha(model, sensor_grid, alphas, A, k_c_values, k_min, include_d2=True):
    """
    Per un alpha fissato, fitta F(k_c) come combinazione lineare di
    n_k(k_c), dn_k/dk(k_c), [d2n_k/dk2(k_c)], con intercetta.
    """

    k_c_values = np.asarray(k_c_values)

    n_vals = A * k_c_values ** (-alphas)
    dn_vals = -alphas * A * k_c_values ** (-alphas - 1)
    d2n_vals = alphas * (alphas + 1) * A * k_c_values ** (-alphas - 2)

    F_true = A * (k_c_values ** (-alphas) - k_min ** (-alphas))

    alphas = np.asarray(alphas)
    profiles = np.array([A * sensor_grid ** (-a) for a in alphas])
    profiles_t = torch.tensor(profiles, dtype=torch.float32)
    k_c_t = torch.full((len(alphas), 1), k_c_values, dtype=torch.float32)

    with torch.no_grad():
        F_pred = model(profiles_t, k_c_t).squeeze().numpy()

    features = [n_vals, dn_vals]
    if include_d2:
        features.append(d2n_vals)
    X = np.stack(features, axis=1)

    reg = LinearRegression(fit_intercept=True)  # <-- intercetta essenziale
    reg.fit(X, F_pred)
    #reg.fit(X, F_pred)

    return reg.coef_, reg.intercept_

def plot_coefficients_vs_alpha(model, sensor_grid, alphas, A, k_c_values, k_min, save_path=None, show=True):
    a_vals, b_vals, c_vals = [], [], []

    for alphas in alphas:
        coefs = fit_operators_at_fixed_alpha(model, sensor_grid, alphas, A, k_c_values, k_min)
        a_vals.append(coefs[0])
        b_vals.append(coefs[1])
        c_vals.append(coefs[2])

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, a_vals, "o-", label=r"$a(\alpha)$ — coeff. su $n_k$")
    plt.plot(alphas, b_vals, "o-", label=r"$b(\alpha)$ — coeff. su $\partial_k n_k$")
    plt.plot(alphas, c_vals, "o-", label=r"$c(\alpha)$ — coeff. su $\partial_k^2 n_k$")
    plt.axhline(0, color="black", linewidth=0.6)
    plt.axhline(1, color="gray", linewidth=0.6, linestyle=":")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("coefficiente")
    plt.title("Coefficienti del fit locale, in funzione di $\\alpha$")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()