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

def plot_flux_vs_k(model, alphas, A, k_min, k_max, sensor_grid, k_c_values, alpha_highlight= -5/3, save_path=None, show=True):
    """
    Mostra il flusso F in funzione di k_c, per uno o più profili n_k
    (uno per ciascun alpha in alphas), sovrapposti.
    """
    k_c_values = np.asarray(k_c_values)
    k_c_t = torch.tensor(k_c_values, dtype=torch.float32).reshape(-1, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for a in alphas:
        profile = A * sensor_grid ** (-a)
        profile_t = torch.tensor(profile, dtype=torch.float32).unsqueeze(0).repeat(len(k_c_values), 1)
        

        with torch.no_grad():
            F_pred = model(profile_t, k_c_t).squeeze().numpy()

        F_true = A * (k_c_values ** (-a) - k_min ** (-a))

        ax1.plot(k_c_values, F_pred, linewidth=2, label=fr"DeepONet, $\alpha={a}$")
        ax1.plot(k_c_values, F_true, "--", linewidth=1.5, label=fr"analitica, $\alpha={a}$")

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel(r"$k_c$")
    ax1.set_ylabel(r"flusso $F$")
    ax1.set_title("Flusso in funzione di $k_c$, per diversi profili spettrali")
    ax1.legend(loc="center left", bbox_to_anchor=(0.8, 0.4))

    profile_h = A * sensor_grid ** (-alpha_highlight)
    profile_h_t = torch.tensor(profile_h, dtype=torch.float32).unsqueeze(0).repeat(len(k_c_values), 1)

    with torch.no_grad():
        F_pred_h = model(profile_h_t, k_c_t).squeeze().numpy()

    F_true_h = A * (k_c_values ** (-alpha_highlight) - k_min ** (-alpha_highlight))

    ax2.plot(k_c_values, F_pred_h, linewidth=2, color="steelblue", label="DeepONet")
    ax2.plot(k_c_values, F_true_h, "--", linewidth=2, color="darkorange", label="analitica")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel(r"$k_c$")
    ax2.set_ylabel(r"flusso $F$")
    ax2.set_title(fr"Flusso vs $k_c$, $\alpha={alpha_highlight:.3f}$")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()