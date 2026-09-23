import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from src.physics.mmt_solver import power_law, flux_ab

def split_deeponet_dataset_TWS(data, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42):
    n = data["profiles"].shape[0]
    idx = torch.arange(n)

    idx_train, idx_temp = train_test_split(idx, train_size=train_frac, random_state=seed)
    rel_val = val_frac / (val_frac + test_frac)
    idx_val, idx_test = train_test_split(idx_temp, train_size=rel_val, random_state=seed)

    def subset(indices):
        return {
            "profiles": data["profiles"][indices],
            "k_c": data["k_c"][indices],
            "p": data["p"][indices], 
        }

    return {"train": subset(idx_train), "val": subset(idx_val), "test": subset(idx_test)}

def train_deeponet_TWS(model, splits, epochs= 6000, lr = 1e-3, device = "cpu", 
                       sensor_grid = None, A = 1.0, k_c_min = 0.1, k_c_max = 1.0,
                       alpha_kz = 0.5, beta = 0.25, lambda_bc = 1.0, n_bc_points = 100):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.MSELoss()

    train_data = splits["train"]
    val_data = splits["val"]

    #-----vincoli fisici da inserire nella loss-----
    p1 = 1 #inserire il valore corretto degli esponenti per le sol di KZ
    p2 = 1

    kz_profile1 = power_law(p1, A, sensor_grid)   #queste sono le soluzioni di KZ
    kz_profile2 = power_law(p2, A, sensor_grid)   
    kz_profile_t1 = torch.tensor(kz_profile1, dtype=torch.float32).unsqueeze(0).repeat(n_bc_points, 1)
    kz_profile_t2 = torch.tensor(kz_profile2, dtype=torch.float32).unsqueeze(0).repeat(n_bc_points, 1)

    k_c_bc = torch.linspace(k_c_min, k_c_max, n_bc_points).reshape(-1, 1)

    F_bc_target = torch.zeros(n_bc_points, 1)

    # qui aggiungeremo un nuovo vincolo per il segno del flusso 

    #-----fine vincoli fisici-----

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        F_pred = model(train_data["profiles"], train_data["k_c"])

        #inserisci i vincoli fisici come loss functions
        F_bc_pred = model()
        bc_loss = loss_fn(F_bc_pred, F_bc_target)

        loss = bc_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            F_val_pred = model(val_data["profiles"], val_data["k_c"])
            val_loss = loss_fn(F_val_pred, val_data["F"])

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 400 == 0 or epoch == epochs - 1:
                print( f"Epoch {epoch:5d} | total_loss: {loss.item():.6e} | " f"val_loss: {val_loss.item():.6e}")

    model.load_state_dict(best_state)
    return model, history

def plot_deeponet_loss(history, save_path=None, show=True):
    plt.figure(figsize=(7, 5))
    plt.semilogy(history["train_loss"], label="train loss")
    plt.semilogy(history["val_loss"], label="validation loss")
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

def plot_flux_prediction_test(model, test_data, save_path=None, show=True):
    model.eval()
    with torch.no_grad():
        F_pred = model(test_data["profiles"], test_data["k_c"])

    plt.figure(figsize=(6, 6))
    plt.scatter(test_data["F"].numpy(), F_pred.numpy(), alpha=0.5, s=15)
    lims = [test_data["F"].min().item(), test_data["F"].max().item()]
    plt.plot(lims, lims, "r--", linewidth=1, label="predizione perfetta")
    plt.xlabel("F teorico")
    plt.ylabel("F predetto (DeepONet)")
    plt.legend()
    plt.title("DeepONet - confronto predizione vs teorico (test set)")
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