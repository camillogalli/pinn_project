import torch

from src.physics.residuals import pde_residual

def compute_loss(model, t_ic, k_ic, n_ic, t_f, k_f, c):
    # 1. Loss sulla condizione iniziale (n_k(0, k) = f(k))
    n_pred_ic = model(torch.cat([t_ic, k_ic], dim=1))
    loss_ic = torch.mean((n_pred_ic - n_ic)**2)

    # 2. Loss sul residuo fisico (nei punti di collocazione)
    residual = pde_residual(model, t_f, k_f, c)
    loss_physics = torch.mean(residual**2)

    # peso relativo (spesso serve bilanciare, es. lambda_ic > 1)
    loss = loss_ic + loss_physics
    return loss, loss_ic, loss_physics