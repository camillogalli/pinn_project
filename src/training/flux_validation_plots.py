import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_flux_vs_alpha(model, alphas, A, k_min, k_max, sensor_grid, k_c, kz_lines=None, save_path=None, show=True):
    """
    Riproduce il tipo di plot di Salvatore et al. Fig.3: flusso F in funzione
    della pendenza spettrale alpha, per un k_c fissato.

    kz_lines: dizionario opzionale {label: alpha_value} per marcare le
    soluzioni note (es. KZ energy, KZ action, equipartizione).
    """
    alphas = np.asarray(alphas)
    profiles = np.array([A * sensor_grid ** (-a) for a in alphas])
    profiles_t = torch.tensor(profiles, dtype=torch.float32)
    k_c_t = torch.full((len(alphas), 1), k_c, dtype=torch.float32)

    with torch.no_grad():
        F_pred = model(profiles_t, k_c_t).squeeze().numpy()

    F_true = A * (k_c ** (-alphas) - k_min ** (-alphas))  # formula esatta

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, F_pred, linewidth=2, label="DeepONet")
    plt.plot(alphas, F_true, "--", linewidth=2, label="analitica")
    plt.axhline(0, color="black", linewidth=0.8)

    if kz_lines:
        for label, a_val in kz_lines.items():
            plt.axvline(a_val, linestyle="--", linewidth=1, label=label)

    plt.xlabel(r"pendenza spettrale $\alpha$")
    plt.ylabel(r"flusso $F$")
    plt.yscale("symlog", linthresh=1.0)  # lineare vicino a zero, log altrove
    plt.title(f"Flusso: DeepONet vs analitico ($k_c={k_c}$)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()

def plot_flux_vs_k(model, alphas, A, k_min, k_max, sensor_grid, k_c_values, save_path=None, show=True):
    """
    Mostra il flusso F in funzione di k_c, per uno o più profili n_k
    (uno per ciascun alpha in alphas), sovrapposti.
    """
    k_c_values = np.asarray(k_c_values)

    plt.figure(figsize=(8, 5))

    for a in alphas:
        profile = A * sensor_grid ** (-a)
        profile_t = torch.tensor(profile, dtype=torch.float32).unsqueeze(0).repeat(len(k_c_values), 1)
        k_c_t = torch.tensor(k_c_values, dtype=torch.float32).reshape(-1, 1)

        with torch.no_grad():
            F_pred = model(profile_t, k_c_t).squeeze().numpy()

        F_true = A * (k_c_values ** (-a) - k_min ** (-a))

        plt.plot(k_c_values, F_pred, linewidth=2, label=fr"DeepONet, $\alpha={a}$")
        plt.plot(k_c_values, F_true, "--", linewidth=1.5, label=fr"analitica, $\alpha={a}$")

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel(r"$k_c$")
    plt.ylabel(r"flusso $F$")
    plt.title("Flusso in funzione di $k_c$, per diversi profili spettrali")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()