from src.models.deeponet import DeepONet
from src.data.deeponet_dataset import build_deeponet_dataset
from src.training.train_deeponet import split_deeponet_dataset, train_deeponet, plot_deeponet_loss, plot_flux_prediction_test

N_SENSORS = 50

data = build_deeponet_dataset(
    alpha_min=1.0, alpha_max=2.0, n_samples=2000, A=1.0,
    k_min=0.1, k_max=1.0, n_sensors=N_SENSORS,
    k_c_min=0.2, k_c_max=0.9
)

splits = split_deeponet_dataset(data, train_frac=0.6, val_frac=0.2, test_frac=0.2)

model = DeepONet(n_sensors=N_SENSORS, latent_dim=32, hidden=64)
model, history = train_deeponet(model, splits, epochs=3000, lr=1e-3)

plot_deeponet_loss(history, save_path="outputs/deeponet_loss.png", show=False)
plot_flux_prediction_test(model, splits["test"], save_path="outputs/deeponet_test.png", show=False)