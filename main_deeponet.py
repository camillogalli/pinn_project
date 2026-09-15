import torch
import numpy as np

from src.models.deeponet import DeepONet
from src.data.deeponet_dataset import build_deeponet_dataset
from src.training.train_deeponet import (
    split_deeponet_dataset, train_deeponet, plot_deeponet_loss, plot_flux_prediction_test)
from src.training.deeponet_analysis import (validate_dF_dkc, plot_dF_dkc_comparison, compute_sensitivity_kernel,
                                             plot_sensitivity_kernel, smooth_kernel)
from src.training.flux_validation_plots import (plot_flux_vs_alpha, plot_flux_vs_k)
from src.utils import config
from src.physics.operator_library import fit_operators_at_fixed_alpha, plot_coefficients_vs_alpha

N_SENSORS = 200

data = build_deeponet_dataset(
    alpha_min=-3, alpha_max=3.0, n_samples=25000, A=1.0,   # range ampliato, include 0
    k_min=config.K_MIN, k_max=config.K_MAX, n_sensors=N_SENSORS,
    k_c_min=0.1, k_c_max=1.0, noise_std=0.01
)

splits = split_deeponet_dataset(data, train_frac=0.6, val_frac=0.2, test_frac=0.2)

model = DeepONet(n_sensors=N_SENSORS, latent_dim=64, hidden=128)

#model, history = train_deeponet(model, splits, epochs=6000, lr=1e-3)
model, history = train_deeponet(
    model, splits, epochs=6000, lr=1e-3,
    sensor_grid=data["sensor_grid"], A=1.0,
    k_c_min=0.1, k_c_max=1.0, lambda_bc=1.0
)

plot_deeponet_loss(history, save_path="outputs/deeponet_loss.png", show=False)
plot_flux_prediction_test(model, splits["test"], save_path="outputs/deeponet_test.png", show=False)

# --- Analisi: validazione dF/dk_c ---
test_data = splits["test"]

# servono anche gli alpha corrispondenti al test set, per il confronto analitico —
# build_deeponet_dataset non li restituisce ancora esplicitamente
dF_learned, dF_true = validate_dF_dkc(
    model, test_data["profiles"], test_data["alphas"], test_data["k_c"], A=1.0
)

plot_dF_dkc_comparison(dF_learned, dF_true, save_path="outputs/dF_dkc_validation.png", show=False)

k_c_fixed = 0.5
profile_example = test_data["profiles"][0]  # un singolo profilo dal test set

#kernel = compute_sensitivity_kernel(model, profile_example, k_c_value=k_c_fixed)
#kernel_smoothed = smooth_kernel(kernel, window=5)

#plot_sensitivity_kernel(
#    torch.tensor(kernel_smoothed), data["sensor_grid"], k_c_value=k_c_fixed, k_min=0.1,
#    save_path="outputs/sensitivity_kernel.png", show=False
#)

plot_flux_vs_alpha(
    model, alphas=np.linspace(-1.5, 1.5, 100), A=1.0,
    k_min=config.K_MIN, k_max=config.K_MAX, sensor_grid=data["sensor_grid"],
    k_c=0.5, kz_lines={"equipartizione (α=0)": 0.0},
    save_path="outputs/flux_vs_alpha.png", show=False
)

plot_flux_vs_k(
    model, alphas=[0.5, 1.0, 1.5, 2.0], A=1.0,
    k_min=config.K_MIN, k_max=config.K_MAX, sensor_grid=data["sensor_grid"],
    k_c_values=np.linspace(config.K_MIN + 0.05, config.K_MAX - 0.05, 100),
    alpha_highlight=5/3,
    save_path="outputs/flux_vs_k.png", show=False
)

plot_coefficients_vs_alpha(
    alphas=[0.5, 1.0, 1.5, 2.0], A = 1.0,
    k_c_values=np.linspace(config.K_MIN + 0.05, config.K_MAX - 0.05, 100),
    k_min=config.K_MIN, save_path="outputs/coefficients_vs_alpha.png", show=False
)