import torch
import numpy as np

from src.models.deeponet import DeepONet
from src.data.deeponet_dataset import build_deeponet_dataset_mmt
from src.training.train_deeponet_mmt import (split_deeponet_dataset_mmt, train_deeponet_mmt)
from src.training.train_deeponet import (
     plot_deeponet_loss, plot_flux_prediction_test)

from src.training.flux_validation_plots import plot_flux_vs_p_mmt
from src.physics.operator_library_mmt import plot_coefficients_vs_p, fit_operators_at_fixed_p, plot_flux_reconstructed
from src.utils import config
from src.physics.linear_operator_library_mmt import fit_operators_at_fixed_p_linear, plot_coefficients_vs_p_linear

N_SENSORS = 200

#data = build_deeponet_dataset_mmt(n_samples = 20000, n_sensors=N_SENSORS, k_c_min=0.1, k_c_max=1.0,
#                                  k_min=config.K_MIN, k_max=config.K_MAX, A=1.0, beta=0.25, N=900, IntN=600,
#                                  noise_std=0.005, seed=42, p_min=-1, p_max=0.06, save_path="mmt_dataset/flux.pt")

data = torch.load("mmt_dataset/flux.pt", weights_only=False)
splits = split_deeponet_dataset_mmt(data, train_frac=0.6, val_frac=0.2, test_frac=0.2)

#model = DeepONet(n_sensors=N_SENSORS, latent_dim=64, hidden=128)

#model, history = train_deeponet_mmt(
#    model, splits, epochs=8000, lr=1e-3,
#    sensor_grid=data["sensor_grid"], alpha_kz=0.5, A=1.0, beta=0.25,
#    k_c_min=0.1, k_c_max=1.0, lambda_bc=1.0, n_bc_points=100
#)
#torch.save(model.state_dict(), "weights/deeponet_weights_mmt.pt")

model = DeepONet(n_sensors=N_SENSORS, latent_dim=64, hidden=128)
model.load_state_dict(torch.load("weights/deeponet_weights_mmt.pt"))
model.eval()

#plot_deeponet_loss(history, save_path="outputs_mmt/deeponet_loss_mmt.png", show=False)
#plot_flux_prediction_test(model, splits["test"], save_path="outputs_mmt/deeponet_test_mmt.png", show=False)

test_data = splits["test"]

#k_c_fixed = 0.5
#profile_example = test_data["profiles"][0]

#plot_flux_vs_p_mmt(
#    model, ps=np.linspace(-0.9, 0.05, 100), A=1.0, beta=0.25, k_c = 1,
#    k_min=config.K_MIN, sensor_grid=data["sensor_grid"],
#    N = 900, IntN = 600, ps_true = np.linspace(-0.9, 0.05, 200),
#    save_path="outputs_mmt/flux_vs_p_mmt.png", show=False
#)

(coef_p, int_p, r2_p), (coef_e, int_e, r2_e) = fit_operators_at_fixed_p(
    model, sensor_grid=data["sensor_grid"], p=-5/6, A=1, 
    k_c_values=np.linspace(0.1, 1.0, 50), beta=0.25, N=900, IntN=600, max_order=3,
    ridge_alpha=1e-3)

plot_coefficients_vs_p(
    model, sensor_grid=data["sensor_grid"], ps=np.linspace(-0.9, 0.05, 50), A=1.0,
    k_c_values=np.linspace(0.1, 1.0, 50),beta=0.25, N=300, IntN=200, max_order=3,
    save_path="outputs_mmt/coefficients_vs_p_mmt.png", show=False
)

#print("Coefficienti (DeepONet):", coef_p, "| intercetta:", int_p, "| R^2:", r2_p)
#print("Coefficienti (solver MMT):", coef_e, "| intercetta:", int_e, "| R^2:", r2_e)

#plot_flux_reconstructed(model, sensor_grid=data["sensor_grid"], k_c_values=np.linspace(0.15, 1.0, 50),
#                        p=-5/6, A=1, beta=0.25, N=900, IntN=600, max_order=3, 
#                        save_path="outputs_mmt/reconstructed_flux_mmt.png", show=False)

#(coef_e_l, int_e_l, r2_e_l, rel_e_l) = fit_operators_at_fixed_p_linear(
#    p = -2/3, A = 1, k = np.linspace(0.1, 1, 50), k_min= config.K_MIN,
#    k_max = config.K_MAX, beta=0.25, N=300, IntN=200, max_order=3, ridge_alpha = 1e-3
#)

#print("Coefficienti (solver MMT): ", coef_e_l, "intercetta: ", int_e_l ,"| R^2: ", r2_e_l, "errore relativo: ", rel_e_l)

#plot_coefficients_vs_p_linear(
#    ps = np.linspace(-0.9, -0.3, 50), A = 1, k = np.linspace(0.1, 1.0, 50), beta = 0.25, 
#    k_min = config.K_MIN, k_max = config.K_MAX, N=300, IntN=200, max_order=3,
#    save_path="outputs_mmt/coefficients_vs_p_mmt_linear", show=False 
#)
