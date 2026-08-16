import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def compare_nested_models(library, F_values, operator_order):
    """
    Confronta modelli annidati di complessità crescente, aggiungendo un
    operatore differenziale alla volta (in ordine di grado), e misura
    il miglioramento del fit (R^2) ad ogni aggiunta.

    operator_order: lista di nomi operatori, in ordine di grado crescente,
    es. ["n_kmax", "dn_dk_kmax", "d2n_dk2_kmax", "d3n_dk3_kmax"]
    """
    y = F_values.numpy().ravel() if torch.is_tensor(F_values) else np.asarray(F_values).ravel()

    results = []
    for i in range(1, len(operator_order) + 1):
        subset = operator_order[:i]
        X = torch.cat([library[name] for name in subset], dim=1).numpy()
        ols = LinearRegression(fit_intercept=False).fit(X, y)
        r2 = ols.score(X, y)
        results.append({"operators": subset, "coefficients": ols.coef_, "r2": r2})
        print(f"Operatori: {subset} -> R^2 = {r2:.6f}, coeff = {ols.coef_}")

    return results


def plot_r2_vs_complexity(results, save_path=None, show=True):
    """Mostra come R^2 migliora aggiungendo operatori via via più complessi."""
    labels = [r["operators"][-1] for r in results]  # ultimo operatore aggiunto ad ogni step
    r2_values = [r["r2"] for r in results]

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(r2_values) + 1), r2_values, "o-", linewidth=2)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=30, ha="right")
    plt.xlabel("operatore aggiunto")
    plt.ylabel(r"$R^2$")
    plt.title("Miglioramento del fit aggiungendo operatori di ordine crescente")
    plt.ylim(0, 1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_final_coefficients(results, save_path=None, show=True):
    """Bar plot dei coefficienti del modello più completo (ultimo della lista)."""
    best = results[-1]
    plt.figure(figsize=(7, 5))
    plt.bar(best["operators"], best["coefficients"], color="steelblue")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("operatore differenziale")
    plt.ylabel("coefficiente (scala fisica)")
    plt.title("Coefficienti del modello completo")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()