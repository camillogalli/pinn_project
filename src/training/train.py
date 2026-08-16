import torch
import matplotlib.pyplot as plt
import numpy as np

from src.training.loss import compute_loss
from src.utils.sampling import sample_points
from src.physics.initial_condition import initial_condition_log
from src.physics.flux import compute_flux


def train(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    loss_history = []
    loss_ic_history = []
    loss_physics_history = []

    for epoch in range(config.EPOCHS):
        t_f, k_f, t_ic, k_ic, n_ic_val = sample_points(
            n_collocation=config.N_COLLOCATION,
            n_ic=config.N_BOUNDARY,
            t_max=config.T_MAX,
            k_min=config.K_MIN,
            k_max=config.K_MAX,
            device=config.DEVICE,
        )

        optimizer.zero_grad()
        loss, loss_ic, loss_physics = compute_loss(
            model, t_ic, k_ic, n_ic_val, t_f, k_f, config.C
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        loss_ic_history.append(loss_ic.item())
        loss_physics_history.append(loss_physics.item())

        if epoch % 200 == 0 or epoch == config.EPOCHS - 1:
            print(
                f"Epoch {epoch:5d} | "
                f"loss: {loss.item():.6e} | "
                f"loss_ic: {loss_ic.item():.6e} | "
                f"loss_physics: {loss_physics.item():.6e}"
            )

    history = {
        "loss": loss_history,
        "loss_ic": loss_ic_history,
        "loss_physics": loss_physics_history,
    }
    return model, history

def moving_average(data, window=50):
    """Media mobile semplice per smoothing visivo della loss."""
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_loss(history, save_path=None, smooth_window=50, show=True):
    plt.figure(figsize=(7, 5))

    for key, label in [
        ("loss", "loss totale"),
        ("loss_ic", "loss IC"),
        ("loss_physics", "loss physics"),
    ]:
        smoothed = moving_average(history[key], window=smooth_window)
        plt.semilogy(smoothed, label=label)

    plt.xlabel("epoca")
    plt.ylabel("loss (scala log, media mobile)")
    plt.legend()
    plt.title("Andamento della loss durante il training (smoothed)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_solution_vs_analytical(model, config, t_snapshot=0.5, n_points=200, save_path=None, show=True):
    """
    Confronta la soluzione predetta dalla rete con la soluzione analitica
    n_k(t, k) = n_k(0, k + c*t)  (traslazione del profilo iniziale)
    """
    k = torch.linspace(config.K_MIN, config.K_MAX, n_points).reshape(-1, 1)
    t = torch.full_like(k, t_snapshot)

    with torch.no_grad():
        log_n_pred = model(torch.cat([t, k], dim=1)).squeeze()
        n_pred = torch.exp(log_n_pred)

    # soluzione analitica: il profilo iniziale traslato di c*t
    k_shifted = k.squeeze() + config.C * t_snapshot
    log_n_analytical = initial_condition_log(k_shifted.clamp(min=config.K_MIN))
    n_analytical = torch.exp(log_n_analytical)

    plt.figure(figsize=(7, 5))
    plt.plot(k.squeeze(), n_pred, label="PINN", linewidth=2)
    plt.plot(k.squeeze(), n_analytical, "--", label="analitica", linewidth=2)
    plt.xlabel("k")
    plt.ylabel("n_k")
    plt.yscale("log")
    plt.legend()
    plt.title(f"Confronto soluzione a t = {t_snapshot}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()