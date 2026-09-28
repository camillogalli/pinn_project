import torch
import matplotlib.pyplot as plt
import numpy as np
from src.physics.mmt_solver import power_law


def sample_homogeneity_batch(n_points, sensor_grid,
                              p_range, A_range, lambda_range, rng):
    p = rng.uniform(*p_range, size=n_points)
    A_amp = rng.uniform(*A_range, size=n_points)
    lam = rng.uniform(*lambda_range, size=n_points)

    profiles = np.array([power_law(pi, Ai, sensor_grid) for pi, Ai in zip(p, A_amp)])
    profiles_scaled = profiles * lam.reshape(-1, 1)

    return (torch.tensor(profiles, dtype=torch.float32),
            torch.tensor(profiles_scaled, dtype=torch.float32),
            torch.tensor(lam, dtype=torch.float32).reshape(-1, 1))

def train_deeponet_TWS(model, epochs= 6000, lr = 1e-3, device = "cpu", 
                       sensor_grid = None, A = 1.0, k_c_min = 0.1, k_c_max = 1.0,
                       d  = 1.0, beta = 0.25, n_bc_points = 100, rng = None, lambda1 = 1, lambda2 = 1, lambda3 = 1):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.MSELoss()


    #-----vincoli fisici da inserire nella loss-----
    p1 = -beta -d #inserire il valore corretto degli esponenti per le sol di KZ
    scaling_factor = 2
    p_sign_min, p_sign_max = p1, 0.0   # regione dove ti aspetti cascata diretta (F>0)
    margin = 0.1                        # IMPORTANTE, vedi sotto
    
    p_sign = rng.uniform(p_sign_min, p_sign_max, size=n_bc_points)
    sign_profiles = np.array([power_law(p, A, sensor_grid) for p in p_sign])
    sign_profiles_t = torch.tensor(sign_profiles, dtype=torch.float32)

    kz_profile1 = power_law(p1, A, sensor_grid)   #queste sono le soluzioni di KZ  
    kz_profile_t1 = torch.tensor(kz_profile1, dtype=torch.float32).unsqueeze(0).repeat(n_bc_points, 1)

    k_c_bc = torch.linspace(k_c_min, k_c_max, n_bc_points).reshape(-1, 1)

    F_bc_target = torch.zeros(n_bc_points, 1)

    # qui aggiungeremo un nuovo vincolo per il segno del flusso 

    #-----fine vincoli fisici-----

    history = {"train_loss": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        #inserisci i vincoli fisici come loss functions

        profiles_raw, profiles_scaled, lam = sample_homogeneity_batch(n_bc_points, sensor_grid,
                                                p_range=(-0.95, 0.05), A_range=(0.5, 2.0), lambda_range=(0.5, 2.0), rng=rng)

        F_bc_pred = model(kz_profile_t1, k_c_bc)
        F_raw = model(profiles_raw, k_c_bc)
        F_scaled = model(profiles_scaled, k_c_bc)
        F_sign_pred = model(sign_profiles_t, k_c_bc)

        sign_loss = torch.relu(margin - F_sign_pred).mean()
        bc_loss1 = loss_fn(F_scaled, (lam ** scaling_factor) * F_raw)
        bc_loss2 = loss_fn(F_bc_pred, F_bc_target)
        bc_loss = lambda1*bc_loss1 + lambda2*bc_loss2 + lambda3*sign_loss

        loss = bc_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        model.eval()

        history["train_loss"].append(loss.item())

        if epoch % 400 == 0 or epoch == epochs - 1:
                print( f"Epoch {epoch:5d} | total_loss: {loss.item():.6e} ")

    return model, history

def plot_deeponet_loss(history, save_path=None, show=True):
    plt.figure(figsize=(7, 5))
    plt.semilogy(history["train_loss"], label="physics loss")
    plt.xlabel("epoca")
    plt.ylabel("MSE (scala log)")
    plt.legend()
    plt.title("DeepONet - andamento della loss - TWS")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_flux_vs_p_TWS(model, ps, A, beta, k_min, sensor_grid, k_c,
                         save_path=None, show=True):
    
    ps = np.asarray(ps)
    profiles = power_law(ps.reshape(-1, 1), A, sensor_grid.reshape(1, -1))
    profiles_t = torch.tensor(profiles, dtype=torch.float32)
    k_c_t = torch.full((len(ps), 1), k_c, dtype=torch.float32)

    with torch.no_grad():
        F_pred = model(profiles_t, k_c_t).squeeze().numpy()

    plt.figure(figsize=(12, 5))
    plt.plot(ps, F_pred, linewidth=3, label="DeepONet", color = "darkred")
    plt.grid(color="lightgrey", linestyle="-", linewidth=0.4)
    plt.axhline(0, color="black", linewidth=0.8)


    plt.xlabel(r"Spectral Power $p$")
    plt.ylabel(r"$^1 P^4_1$")
    plt.ylim(-5, 15)
    plt.xlim(-0.9, 0.15)
    plt.title(r"Wave action flux: DeepONet vs solver MMT ($^1 P^4_{(0, 1) \rightarrow (1, \infty)}$)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()