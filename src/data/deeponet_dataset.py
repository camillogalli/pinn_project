import numpy as np
import torch


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


def build_deeponet_dataset(alpha_min, alpha_max, n_samples, A, k_min, k_max,
                            n_sensors, k_c_min, k_c_max, noise_std=0.01, seed=42):
    rng = np.random.default_rng(seed)
    alphas = rng.uniform(alpha_min, alpha_max, size=n_samples)
    k_cs = rng.beta(a=0.5, b=2.0, size=n_samples) * (k_c_max - k_c_min) + k_c_min

    sensor_grid = sample_sensor_grid(k_min, k_max, n_sensors)

    # PRIMA: profiles = generate_power_law_profiles(alphas, A, sensor_grid)
    profiles = generate_power_law_profiles_perturbed(alphas, A, sensor_grid, noise_std=noise_std)

    F_values = compute_true_flux(alphas, A, k_cs, k_min)

    return {
        "profiles": torch.tensor(profiles, dtype=torch.float32),
        "k_c": torch.tensor(k_cs, dtype=torch.float32).reshape(-1, 1),
        "F": torch.tensor(F_values, dtype=torch.float32).reshape(-1, 1),
        "alphas": torch.tensor(alphas, dtype=torch.float32).reshape(-1, 1),
        "sensor_grid": sensor_grid,
    }

def generate_power_law_profiles_perturbed(alphas, A, sensor_grid, noise_std=0.01):
    profiles = np.array([A * sensor_grid ** (-a) for a in alphas])
    noise = np.random.normal(0, noise_std * np.abs(profiles))
    return profiles + noise