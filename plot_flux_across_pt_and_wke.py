"""
Traduzione Python di plot_FluxAcrossPtandWKE.m — riproduce la Fig. 3 di
Salvatore et al.: flusso di energia e azione attraverso k_c=1, in funzione
della pendenza spettrale p, per spettri power-law n_k = k^p.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.physics.mmt_solver import power_law, flux_ab


def run(alpha=0.5, beta=0.25, N=800, IntN=500, n_points=60, save_path=None, show=True):
    # Soluzioni KZ analitiche (per riferimento sul plot)
    eq1 = (2 * beta / (3 * alpha)) - 1 / alpha  # Energy KZ (in spazio omega)
    eq2 = eq1 + 1 / 3                            # Action KZ
    eq1K, eq2K = eq1 / 2, eq2 / 2                 # convertiti in spazio k

    p_values = np.linspace(eq1K - 0.05, 0.05, n_points)

    F_energy = np.zeros(n_points)
    F_action = np.zeros(n_points)
    deriv = np.zeros(n_points)   # "WKE at k=1": derivata del flusso di azione rispetto a k_c, in k_c=1

    dk = 0.01

    for i, p in enumerate(p_values):
        n_params = (p, 1.0)

        F_energy[i] = flux_ab(power_law, n_params, [0, 1], [1, np.inf], beta, "Energy", N=N, IntN=IntN)[0]
        F_action[i] = flux_ab(power_law, n_params, [0, 1], [1, np.inf], beta, "Action", N=N, IntN=IntN)[0]

        # derivata numerica del flusso d'azione rispetto a k_c, attorno a k_c=1
        A_in = [[0, 1 + dk], [0, 1 - dk]]
        B_in = [[1 + dk, np.inf], [1 - dk, np.inf]]
        F_shift = flux_ab(power_law, n_params, A_in, B_in, beta, "Action", N=N, IntN=IntN)
        deriv[i] = (F_shift[0] - F_shift[1]) / (2 * dk)

        print(f"[{i+1:3d}/{n_points}] p={p:8.4f} | F_energy={F_energy[i]:10.4f} "
              f"| F_action={F_action[i]:10.4f} | WKE(k=1)={deriv[i]:10.4f}")

    # --- Plot, stile Fig. 3 ---
    plt.figure(figsize=(9, 6))
    plt.plot(p_values, F_energy, linewidth=1.5, label=r"$\rho=\omega$, Energy Flux")
    plt.plot(p_values, F_action, "--", linewidth=1.5, label=r"$\rho=1$, Action Flux")
    plt.plot(p_values, deriv, "-.", color="black", linewidth=1, label="WKE at k=1")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(eq1K, color="tab:blue", linestyle=":", linewidth=1)
    plt.axvline(eq2K, color="tab:blue", linestyle=":", linewidth=1)
    plt.text(eq1K, max(F_energy.max(), F_action.max()) * 0.9, "Energy KZ Sol.", rotation=90, va="top")
    plt.text(eq2K, max(F_energy.max(), F_action.max()) * 0.9, "Action KZ Sol.", rotation=90, va="top")
    plt.xlabel("Spectral Power (p)")
    plt.ylabel(r"$^{\rho}P^4_{(0,1)\to(1,\infty)}$")
    plt.title(r"Flux of Energy and Action $(0,1)\to(1,\infty)$ as a Function of Power Law")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return p_values, F_energy, F_action, deriv


if __name__ == "__main__":
    run(save_path="outputs/flux_across_pt_and_wke.png", show=False)
