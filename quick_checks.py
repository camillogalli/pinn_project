import torch
import numpy as np
from src.models.deeponet import DeepONet
from src.data.deeponet_dataset import build_deeponet_dataset
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

k_c_values = np.linspace(0.1, 1.0, 100)
alpha = 0.1  # esempio

n_vals = 1.0 * k_c_values ** (-alpha)
profile = 1.0 * data["sensor_grid"] ** (-alpha)
profile_t = torch.tensor(profile, dtype=torch.float32).unsqueeze(0).repeat(len(k_c_values), 1)
k_c_t = torch.tensor(k_c_values, dtype=torch.float32).reshape(-1, 1)

with torch.no_grad():
    F_pred = model(profile_t, k_c_t).squeeze().numpy()

F_true = 1.0 * (k_c_values ** (-alpha) - config.K_MIN ** (-alpha))

signal_amplitude = F_true.max() - F_true.min()
noise_estimate = np.std(F_pred - F_true)
print(f"alpha={alpha}: segnale={signal_amplitude:.4f}, rumore stimato={noise_estimate:.4f}, " f"SNR={signal_amplitude/noise_estimate:.2f}")