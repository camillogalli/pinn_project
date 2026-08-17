from src.models.deeponet import DeepONet
from src.data.deeponet_dataset import build_deeponet_dataset
from src.training.train_deeponet import (
    split_deeponet_dataset, train_deeponet, plot_deeponet_loss, plot_flux_prediction_test)
from src.training.deeponet_analysis import (validate_dF_dkc, plot_dF_dkc_comparison, compute_sensitivity_kernel, plot_sensitivity_kernel)

N_SENSORS = 50

data = build_deeponet_dataset(
    alpha_min=1.0, alpha_max=2.0, n_samples=5000, A=1.0,
    k_min=0.1, k_max=1.0, n_sensors=N_SENSORS,
    k_c_min=0.2, k_c_max=0.9
)

splits = split_deeponet_dataset(data, train_frac=0.6, val_frac=0.2, test_frac=0.2)

model = DeepONet(n_sensors=N_SENSORS, latent_dim=64, hidden=128)
model, history = train_deeponet(model, splits, epochs=3000, lr=1e-3)

plot_deeponet_loss(history, save_path="outputs/deeponet_loss.png", show=False)
plot_flux_prediction_test(model, splits["test"], save_path="outputs/deeponet_test.png", show=False)

# --- Analisi: validazione dF/dk_c ---
test_data = splits["test"]

# servono anche gli alpha corrispondenti al test set, per il confronto analitico —
# vedi nota sotto: build_deeponet_dataset non li restituisce ancora esplicitamente
dF_learned, dF_true = validate_dF_dkc(
    model, test_data["profiles"], test_data["alphas"], test_data["k_c"], A=1.0
)

plot_dF_dkc_comparison(dF_learned, dF_true, save_path="outputs/dF_dkc_validation.png", show=False)

k_c_fixed = 0.5
profile_example = test_data["profiles"][0]  # un singolo profilo dal test set

kernel = compute_sensitivity_kernel(model, profile_example, k_c_value=k_c_fixed)

plot_sensitivity_kernel(
    kernel, data["sensor_grid"], k_c_value=k_c_fixed, k_min=0.1,
    save_path="outputs/sensitivity_kernel.png", show=False
)