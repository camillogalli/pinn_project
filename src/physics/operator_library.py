import torch


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