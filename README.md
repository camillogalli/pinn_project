# Analyzing Wave Turbulence Fluxes with a Combined PINN/DeepONet Framework

**Author:** Camillo Galli — Thesis project, July 2026

A physics-informed and data-driven approach to wave turbulence theory: this project builds neural network models capable of learning the spectral energy/action flux of a wave kinetic equation (WKE), starting from a simplified transport equation and progressively moving toward the Majda–McLaughlin–Tabak (MMT) four-wave model, with the eventual goal of generalizing to a three-wave interaction kernel.

Two complementary machine learning paradigms are explored and compared:
- **Physics-Informed Neural Networks (PINNs)** — learn a solution field constrained by a known governing PDE.
- **Deep Operator Networks (DeepONets)** — learn the operator mapping a spectral density profile $n_k(\cdot)$ to its associated flux $F$, without requiring the underlying equation to be known in closed form.

## Features

- **Transport equation PINN** — validates the PINN methodology on the linear, analytically solvable equation $\partial_t n_k = c\,\partial_k n_k$, including flux computation and extrapolation-limit analysis.
- **MMT collision integral solver** — a faithful Python translation of the original MATLAB solver for the four-wave MMT wave kinetic equation, validated against known Kolmogorov–Zakharov (KZ) exponents.
- **DeepONet operator learning** — learns the mapping $n_k(\cdot) \to F$ directly from data, first validated on synthetic power-law spectra with a closed-form flux, then trained on the real MMT dataset.
- **Flux structure investigation** — several approaches to recovering an interpretable functional form of the flux (sparse regression, nested nonlinear operator libraries, gradient-based sensitivity), with an honest account of what worked and what didn't.
- **Direct physical validation** — flux-vs-spectral-slope curves reproducing the KZ/equipartition zero-crossings of the reference literature (Salvatore et al.).

---

## Methodology Overview

1. **Transport equation validation** — a PINN is trained on $\partial_t n_k = c\,\partial_k n_k$ with a power-law initial condition, validated against the exact analytical solution and its flux.
2. **MMT dataset generation** — the translated collision integral solver produces $(n_k, k_c) \to F$ samples for a family of power-law spectra.
3. **DeepONet training** — a branch/trunk architecture learns the flux operator from the MMT dataset, validated via parity plots, derivative consistency ($\partial F/\partial k_c$), and direct comparison against the solver across a range of spectral slopes.
4. **Flux structure analysis** — nested linear and cubic operator libraries are fit to the (emulated and true) flux to probe which terms dominate; found to be non-stationary across spectral slopes, motivating a shift toward direct physical validation over explicit functional-form recovery.
5. **Generalization (in progress)** — transfer learning from the MMT-trained DeepONet toward a three-wave interaction kernel, using physics-informed self-supervised constraints (homogeneity/scaling relations, known KZ zeros) in place of a labeled dataset, since no three-wave solver is available.

---

## Requirements

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy scipy matplotlib scikit-learn
pip install -r requirements.txt
```

Developed and tested on Ubuntu (WSL2), CPU-only (no CUDA-capable GPU available).

---

## Usage

```bash
# PINN on the transport equation
python main.py

# DeepONet, synthetic power-law validation
python main_deeponet.py

# DeepONet trained on the real MMT dataset
python main_mmt.py

# Standalone MMT solver validation (reproduces the reference flux-vs-slope figure)
python plot_flux_across_pt_and_wke.py
```

Generated figures are saved to `outputs/`; trained model weights to `weights/`; the cached MMT dataset to `mmt_dataset/`.

---

## Acknowledgements

The MMT collision integral solver is translated from an original MATLAB implementation; the reference flux-vs-spectral-slope figure follows the structure of N.R. Salvatore et al., *Physica D: Nonlinear Phenomena* 483 (2025) 134964.
