import torch
import numpy as np
from src.models.deeponet import DeepONet
from src.training.train_deeponet_TWS import ( train_deeponet_TWS,
                                              plot_deeponet_loss,  plot_flux_vs_p_TWS)
from src.data.deeponet_dataset import sample_sensor_grid
from src.utils import config

N_SENSORS = 200
sensor_grid = sample_sensor_grid(config.K_MIN, config.K_MAX, N_SENSORS)
rng = np.random.default_rng(seed=42)

model = DeepONet(n_sensors=N_SENSORS, latent_dim=64, hidden=128)
model.load_state_dict(torch.load("weights/deeponet_weights_mmt.pt"))

model, history = train_deeponet_TWS(
    model, epochs=6000, lr=1e-3,
    sensor_grid=sensor_grid, A=1.0, k_c_min=0.1, k_c_max=1.0,
    d = 1.0, beta=1, n_bc_points=100, rng = rng
)
model.eval()
torch.save(model.state_dict(), "weights/deeponet_weights_TWS.pt")

plot_deeponet_loss(history, save_path="outputs_TWS/deeponet_loss_TWS.png", show=False)

plot_flux_vs_p_TWS(
    model, ps=np.linspace(-0.9, 0.05, 100), A=1.0, beta=1, 
    k_min=config.K_MIN, sensor_grid=sensor_grid, k_c = 1,
    save_path="outputs_TWS/flux_vs_p_TWS.png", show=False
)