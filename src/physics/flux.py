import torch


def compute_flux(model, t_val, k_min=0.1, k_max=1.0, n_points=500, device="cpu"):
    """
    Calcola il flusso Phi(t) = integral_0^1 (dn/dk) dk
    tramite integrazione numerica dei gradienti della rete.
    """
    k = torch.linspace(k_min, k_max, n_points, requires_grad=True).reshape(-1, 1).to(device)
    t = torch.full_like(k, t_val)

    inputs = torch.cat([t, k], dim=1)

    log_n_k = model(inputs)
    n_k = model(inputs)

    dn_dk = torch.autograd.grad(
        n_k, k,
        grad_outputs=torch.ones_like(n_k),
        create_graph=False
    )[0]

    flux = torch.trapz(dn_dk.squeeze(), k.squeeze())
    return flux.item()