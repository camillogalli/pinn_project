import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def split_deeponet_dataset(data, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42):
    n = data["profiles"].shape[0]
    idx = torch.arange(n)

    idx_train, idx_temp = train_test_split(idx, train_size=train_frac, random_state=seed)
    rel_val = val_frac / (val_frac + test_frac)
    idx_val, idx_test = train_test_split(idx_temp, train_size=rel_val, random_state=seed)

    def subset(indices):
        return {
            "profiles": data["profiles"][indices],
            "k_c": data["k_c"][indices],
            "F": data["F"][indices],
        }

    return {"train": subset(idx_train), "val": subset(idx_val), "test": subset(idx_test)}


def train_deeponet(model, splits, epochs=3000, lr=1e-3, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    loss_fn = torch.nn.MSELoss()

    train_data = splits["train"]
    val_data = splits["val"]

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        F_pred = model(train_data["profiles"], train_data["k_c"])
        loss = loss_fn(F_pred, train_data["F"])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            F_val_pred = model(val_data["profiles"], val_data["k_c"])
            val_loss = loss_fn(F_val_pred, val_data["F"])

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 200 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:5d} | train_loss: {loss.item():.6e} | val_loss: {val_loss.item():.6e}")

    model.load_state_dict(best_state)
    return model, history


def plot_deeponet_loss(history, save_path=None, show=True):
    plt.figure(figsize=(7, 5))
    plt.semilogy(history["train_loss"], label="train loss")
    plt.semilogy(history["val_loss"], label="validation loss")
    plt.xlabel("epoca")
    plt.ylabel("MSE (scala log)")
    plt.legend()
    plt.title("DeepONet — andamento della loss")
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
    plt.title("DeepONet — confronto predizione vs teorico (test set)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()