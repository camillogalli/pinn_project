from matplotlib.style import library

from src.models.pinn import PINN
from src.training.train import train, plot_loss, plot_solution_vs_analytical
from src.physics.flux import compute_flux
from src.physics.operator_library import analytical_operator_library
from src.training.model_comparison import (
    compare_nested_models,
    plot_r2_vs_complexity,
    plot_final_coefficients,
)

import torch
import numpy as np
import src.utils.config as config


def main():
    model = PINN(layers=[2, 20, 20, 20, 1])

    model, history = train(model, config)

    plot_loss(history, save_path="outputs/loss.png", show=False)
    plot_solution_vs_analytical(model, config, t_snapshot=0.5, save_path="outputs/solution.png", show=False)

    flux_t0 = compute_flux(model, t_val=0.0, k_min=config.K_MIN, k_max=config.K_MAX)
    flux_t1 = compute_flux(model, t_val=1.0, k_min=config.K_MIN, k_max=config.K_MAX)
    print(f"Flusso a t=0: {flux_t0:.6e}")
    print(f"Flusso a t=1: {flux_t1:.6e}")

    # --- Flux discovery: validazione della pipeline su soluzioni analitiche ---
    # NB: campioniamo tanti alpha indipendenti (spettri diversi), non punti lungo
    # un'unica traiettoria temporale, per evitare la collinearità delle derivate
    # lungo le caratteristiche (vedi discussione precedente).
    alphas = np.random.uniform(1.0, 2.0, size=200)
    k_c = 0.7  # diverso da K_MAX=1.0, altrimenti n(k_c) non varierebbe con alpha

    library, F_true = analytical_operator_library(alphas, A=1.0, k_max=k_c, k_min=config.K_MIN)

    operator_order = ["n_kmax", "dn_dk_kmax", "d2n_dk2_kmax", "d3n_dk3_kmax",
                  "n_kmin", "dn_dk_kmin", "d2n_dk2_kmin", "d3n_dk3_kmin"]

    results = compare_nested_models(library, F_true, operator_order)

    plot_r2_vs_complexity(results, save_path="outputs/r2_vs_complexity.png", show=False)
    plot_final_coefficients(results, save_path="outputs/final_coefficients.png", show=False)


if __name__ == "__main__":
    main()