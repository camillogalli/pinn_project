import torch
import numpy as np

from src.models.deeponet import DeepONet
from src.data.deeponet_dataset import build_deeponet_dataset_mmt
from src.training.train_deeponet_mmt import (split_deeponet_dataset_mmt, train_deeponet_mmt)
from src.training.train_deeponet import (
     plot_deeponet_loss, plot_flux_prediction_test)
from src.training.deeponet_analysis import (validate_dF_dkc, plot_dF_dkc_comparison)
from src.training.flux_validation_plots import (plot_flux_vs_p_mmt, plot_flux_vs_k_mmt)
from src.utils import config

N_SENSORS = 200

#data = build_deeponet_dataset_mmt(n_samples = 10000, n_sensors=N_SENSORS, k_c_min=0.1, k_c_max=1.0,
#                                  k_min=config.K_MIN, k_max=config.K_MAX, A=1.0, beta=0.25, N=300, IntN=200,
#                                  noise_std=0.01, seed=42, p_min=-2, p_max=0.4, save_path="mmt_dataset/flux.pt")

data = torch.load("mmt_dataset/flux.pt", weights_only=False)

splits = split_deeponet_dataset_mmt(data, train_frac=0.6, val_frac=0.2, test_frac=0.2)

model = DeepONet(n_sensors=N_SENSORS, latent_dim=64, hidden=128)

model, history = train_deeponet_mmt(
    model, splits, epochs=6000, lr=1e-3,
    sensor_grid=data["sensor_grid"], alpha_kz=0.5, A=1.0, beta=0.25,
    k_c_min=0.1, k_c_max=1.0, lambda_bc=1.0, n_bc_points=50
)

plot_deeponet_loss(history, save_path="outputs_mmt/deeponet_loss_mmt.png", show=False)
plot_flux_prediction_test(model, splits["test"], save_path="outputs_mmt/deeponet_test_mmt.png", show=False)

test_data = splits["test"]

dF_learned, dF_true = validate_dF_dkc(
    model, test_data["profiles"], test_data["p"], test_data["k_c"], A=1.0
)

plot_dF_dkc_comparison(dF_learned, dF_true, save_path="outputs_mmt/dF_dkc_validation_mmt.png", show=False)

k_c_fixed = 0.5
profile_example = test_data["profiles"][0]

plot_flux_vs_p_mmt(
    model, ps=np.linspace(-2, 0.4, 100), A=1.0, beta=0.25, k_c = 1,
    k_min=config.K_MIN, sensor_grid=data["sensor_grid"],
    N = 300, IntN = 200, kz_lines = None, ps_true = np.linspace(-2, 0.4, 200),
    save_path="outputs_mmt/flux_vs_p_mmt.png", show=False
)

#plot_flux_vs_k_mmt(
#    model, ps=np.linspace(-2, 0.4, 100), A=1.0, beta=0.25,
#    k_min=config.K_MIN, k_max=config.K_MAX, sensor_grid=data["sensor_grid"],
#    k_c_values=np.linspace(0.1, 1.0, 100), k_c_values_true=None,
#    p_highlight=-5/3, N = 300, IntN = 200,
#    save_path="outputs_mmt/flux_vs_k_mmt.png", show=False
#)