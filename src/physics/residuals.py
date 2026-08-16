import torch

def pde_residual(model, t, k, c):
    """
    Calcola il residuo dell'equazione di trasporto:
    r = dn/dt - c * dn/dk
    
    t, k: tensori con requires_grad=True, shape (N, 1)
    """
    inputs = torch.cat([t, k], dim=1)
    n_k = model(inputs)  # output della rete: n_k(t, k)

    # derivate tramite autograd
    dn_dt = torch.autograd.grad(
        n_k, t,
        grad_outputs=torch.ones_like(n_k),
        create_graph=True
    )[0]

    dn_dk = torch.autograd.grad(
        n_k, k,
        grad_outputs=torch.ones_like(n_k),
        create_graph=True
    )[0]

    residual = dn_dt - c * dn_dk
    return residual