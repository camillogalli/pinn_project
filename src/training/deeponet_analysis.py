import torch
import matplotlib.pyplot as plt
import numpy as np

def validate_dF_dkc(model, profiles, alphas, k_c_values, A=1.0):
    """
    Confronta dF/dk_c (autograd sul DeepONet) con la derivata analitica vera
    dn_k/dk valutata a k_c, per validare che la rete abbia imparato
    correttamente la struttura dell'operatore (non solo i valori di F).
    """
    k_c = k_c_values.clone().requires_grad_(True)
    F_pred = model(profiles, k_c)
    dF_dkc_learned = torch.autograd.grad(F_pred, k_c, grad_outputs=torch.ones_like(F_pred))[0]

    alphas_t = torch.as_tensor(alphas, dtype=torch.float32).reshape(-1, 1)
    dF_dkc_true = -alphas_t * A * k_c.detach() ** (-alphas_t - 1)

    return dF_dkc_learned.detach(), dF_dkc_true

def plot_dF_dkc_comparison(dF_learned, dF_true, save_path=None, show=True):
    plt.figure(figsize=(6, 6))
    plt.scatter(dF_true.numpy(), dF_learned.numpy(), alpha=0.4, s=15)
    lims = [dF_true.min().item(), dF_true.max().item()]
    plt.plot(lims, lims, "r--", linewidth=1, label="atteso")
    plt.xlabel(r"$\partial_k n_k(k_c)$ teorico")
    plt.ylabel(r"$\partial F/\partial k_c$ (DeepONet, autograd)")
    plt.legend()
    plt.title("Validazione: derivata del flusso vs derivata dello spettro")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    if show: plt.show()
    plt.close()

def compute_sensitivity_kernel(model, profile_single, k_c_value):
    """
    Kernel di sensibilità K(k_i) = dF/dn_k(k_i) per UN singolo profilo,
    ad un dato k_c.
    """
    profile = profile_single.clone().unsqueeze(0).requires_grad_(True)  # (1, n_sensors)
    k_c = torch.tensor([[k_c_value]], dtype=torch.float32)

    F_pred = model(profile, k_c)
    kernel = torch.autograd.grad(F_pred, profile)[0]
    return kernel.squeeze()  # (n_sensors,)

def smooth_kernel(kernel, window=5):
    """Media mobile per smussare visivamente il kernel di sensibilità."""
    kernel_np = kernel.detach().numpy()
    return np.convolve(kernel_np, np.ones(window) / window, mode='same')


def plot_sensitivity_kernel(kernel, sensor_grid, k_c_value, k_min=None, save_path=None, show=True):
    plt.figure(figsize=(7, 5))
    plt.plot(sensor_grid, kernel.detach().numpy(), linewidth=2)
    plt.axvline(k_c_value, color="red", linestyle="--", label=r"$k_c$")
    if k_min is not None:
        plt.axvline(k_min, color="green", linestyle="--", label=r"$k_{min}$")
    plt.axhline(0, color="black", linewidth=0.6)
    plt.xlabel("k")
    plt.ylabel(r"$\partial F / \partial n_k(k)$")
    plt.title(f"Kernel di sensibilità del flusso, $k_c={k_c_value}$")
    plt.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    if show: plt.show()
    plt.close()