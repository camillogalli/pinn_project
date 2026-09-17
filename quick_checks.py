import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.deeponet import DeepONet
from src.data.deeponet_dataset import build_deeponet_dataset
from src.physics.operator_library import fit_operators_at_fixed_alpha  # adatta il path se necessario
from src.utils import config

N_SENSORS = 200

data = build_deeponet_dataset(
    alpha_min=-3, alpha_max=3.0, n_samples=25000, A=1.0,
    k_min=config.K_MIN, k_max=config.K_MAX, n_sensors=N_SENSORS,
    k_c_min=0.1, k_c_max=1.0, noise_std=0.01
)

model = DeepONet(n_sensors=N_SENSORS, latent_dim=64, hidden=128)
model.load_state_dict(torch.load("weights/deeponet_weights.pt"))
model.eval()

alphas_test = np.linspace(-1.5, 1.5, 50)  # copre l'intero range che hai usato nei plot precedenti
k_c_values = np.linspace(0.2, 0.9, 100)

snr_values = []
a_values = []

print(f"{'alpha':>8} | {'segnale':>10} | {'rumore':>10} | {'SNR':>10}")
for alpha in alphas_test:
    profile = 1.0 * data["sensor_grid"] ** (-alpha)
    profile_t = torch.tensor(profile, dtype=torch.float32).unsqueeze(0).repeat(len(k_c_values), 1)
    k_c_t = torch.tensor(k_c_values, dtype=torch.float32).reshape(-1, 1)

    with torch.no_grad():
        F_pred = model(profile_t, k_c_t).squeeze().numpy()

    F_true = 1.0 * (k_c_values ** (-alpha) - config.K_MIN ** (-alpha))

    signal_amplitude = F_true.max() - F_true.min()
    noise_estimate = np.std(F_pred - F_true)
    snr = signal_amplitude / (noise_estimate + 1e-12)   # eps per evitare divisione per zero esatto

    print(f"{alpha:8.3f} | {signal_amplitude:10.4f} | {noise_estimate:10.4f} | {snr:10.2f}")

    snr_values.append(snr)

    (coef_p, _), _ = fit_operators_at_fixed_alpha(model, data["sensor_grid"], alpha, 1.0, k_c_values, config.K_MIN)
    a_values.append(coef_p[0])

snr_values = np.array(snr_values)
a_values = np.array(a_values)

# --- Plot combinato, assi doppi ---
fig, ax1 = plt.subplots(figsize=(9, 5))

color_a = "tab:blue"
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel(r"$a(\alpha)$ — coeff. su $n_k$", color=color_a)
ax1.plot(alphas_test, a_values, "o-", color=color_a, markersize=3, label=r"$a(\alpha)$")
ax1.axhline(1, color=color_a, linewidth=0.8, linestyle=":")
ax1.tick_params(axis="y", labelcolor=color_a)

ax2 = ax1.twinx()
color_snr = "tab:red"
ax2.set_ylabel("SNR", color=color_snr)
ax2.plot(alphas_test, snr_values, "s--", color=color_snr, markersize=3, label="SNR")
ax2.set_yscale("log")
ax2.tick_params(axis="y", labelcolor=color_snr)

plt.title(r"Coefficiente $a(\alpha)$ e SNR del segnale $F(k_c)$, a confronto")
fig.tight_layout()
plt.savefig("outputs/snr_vs_coefficient.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()