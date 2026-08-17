import torch
import matplotlib.pyplot as plt

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