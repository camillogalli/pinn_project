import torch
from src.physics.initial_condition import initial_condition_log


def sample_points(n_collocation, n_ic, t_max=1.0, k_min=0.1, k_max=1.0, device="cpu"):
    """
    Genera i punti per il training:
    - punti di collocazione (t_f, k_f) dove viene imposto il residuo PDE
    - punti sulla condizione iniziale (t_ic, k_ic, n_ic_val) dove t=0

    Ritorna tutti i tensori pronti per essere passati a compute_loss.
    """
    # punti di collocazione (dentro il dominio t-k) per il residuo PDE
    t_f = torch.rand(n_collocation, 1, device=device) * t_max
    k_f = torch.rand(n_collocation, 1, device=device) * (k_max - k_min) + k_min
    t_f.requires_grad_(True)
    k_f.requires_grad_(True)

    # punti sulla condizione iniziale (t=0)
    k_ic = torch.rand(n_ic, 1, device=device) * (k_max - k_min) + k_min
    t_ic = torch.zeros_like(k_ic)
    n_ic_val = initial_condition_log(k_ic)

    return t_f, k_f, t_ic, k_ic, n_ic_val
