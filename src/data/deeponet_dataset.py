import numpy as np
import torch
from src.physics.mmt_solver import power_law, flux_ab


def sample_sensor_grid(k_min, k_max, n_sensors):
    """Griglia fissa di punti sensore su cui campionare ogni profilo n_k."""
    return np.linspace(k_min, k_max, n_sensors)


def generate_power_law_profiles(alphas, A, sensor_grid):
    """
    Per ogni alpha, genera il profilo n_k = A * k^-alpha campionato
    sulla griglia di sensori. Ritorna una matrice (n_samples, n_sensors).
    """
    profiles = np.array([A * sensor_grid ** (-a) for a in alphas])
    return profiles


def compute_true_flux(alphas, A, k_c, k_min):
    """Flusso vero (analitico) per ogni alpha, al punto critico k_c."""
    alphas = np.asarray(alphas)
    return A * k_c ** (-alphas) - A * k_min ** (-alphas)

def generate_power_law_profiles_perturbed(alphas, A, sensor_grid, noise_std=0.01, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    profiles = np.array([A * sensor_grid ** (-a) for a in alphas])
    noise = rng.normal(0, noise_std * np.abs(profiles))
    return profiles + noise


def build_deeponet_dataset(alpha_min, alpha_max, n_samples, A, k_min, k_max,
                            n_sensors, k_c_min, k_c_max, noise_std=0.01, seed=42):
    rng = np.random.default_rng(seed)
    alphas = rng.beta(a=2.0, b=2, size=n_samples) * (alpha_max - alpha_min) + alpha_min
    k_cs = rng.beta(a=0.5, b=2.0, size=n_samples) * (k_c_max - k_c_min) + k_c_min

    sensor_grid = sample_sensor_grid(k_min, k_max, n_sensors)

    # PRIMA: profiles = generate_power_law_profiles(alphas, A, sensor_grid)
    profiles = generate_power_law_profiles_perturbed(alphas, A, sensor_grid, noise_std=noise_std, rng=rng)

    F_values = compute_true_flux(alphas, A, k_cs, k_min)

    return {
        "profiles": torch.tensor(profiles, dtype=torch.float32),
        "k_c": torch.tensor(k_cs, dtype=torch.float32).reshape(-1, 1),
        "F": torch.tensor(F_values, dtype=torch.float32).reshape(-1, 1),
        "alphas": torch.tensor(alphas, dtype=torch.float32).reshape(-1, 1),
        "sensor_grid": sensor_grid,
    }

def build_deeponet_dataset_mmt(n_samples, n_sensors, k_c_min, k_c_max, k_min, k_max,
                                A=1.0, beta=0.25, N=300, IntN=200, noise_std=0.01,
                                seed=42, p_min=-2, p_max=0.4, save_path="flux.pt"):

    rng = np.random.default_rng(seed)
    p = rng.beta(a=2.0, b=2.0, size=n_samples) * (p_max - p_min) + p_min
    k_cs = rng.beta(a=0.5, b=2.0, size=n_samples) * (k_c_max - k_c_min) + k_c_min

    sensor_grid = sample_sensor_grid(k_min, k_max, n_sensors)
    type_ = "Action"

    profiles = power_law(p.reshape(-1, 1), A, sensor_grid.reshape(1, -1))
    noise = rng.normal(0, noise_std * np.abs(profiles))
    profiles = profiles + noise

    flux = np.zeros(n_samples)
    for i in range(n_samples):
        n_params = (p[i], A)
        A_in = [0, k_cs[i]]
        B_in = [k_cs[i], np.inf]
        flux[i] = flux_ab(power_law, n_params, A_in, B_in, beta, type_, N=N, IntN=IntN)[0]

        if i % 500 == 0:
            print(f"[{i}/{n_samples}] p={p[i]:.4f} k_c={k_cs[i]:.4f} -> F={flux[i]:.4f}")

    dataset = {
        "profiles": torch.tensor(profiles, dtype=torch.float32),
        "k_c": torch.tensor(k_cs, dtype=torch.float32).reshape(-1, 1),
        "F": torch.tensor(flux, dtype=torch.float32).reshape(-1, 1),
        "p": torch.tensor(p, dtype=torch.float32).reshape(-1, 1),
        "sensor_grid": sensor_grid,
    }

    torch.save(dataset, save_path)
    return dataset
    