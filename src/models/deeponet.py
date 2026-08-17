import torch
import torch.nn as nn


class DeepONet(nn.Module):
    """
    DeepONet per l'apprendimento dell'operatore n_k(.) -> F(k_c).

    Branch net: codifica il profilo n_k campionato su una griglia fissa di sensori.
    Trunk net: codifica la coordinata di valutazione k_c.
    Output: prodotto scalare tra le due rappresentazioni latenti.
    """
    def __init__(self, n_sensors, latent_dim=64, hidden=128):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(n_sensors, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim)
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, n_k_sampled, k_c):
        """
        n_k_sampled: (batch, n_sensors) — profilo campionato sui sensori
        k_c: (batch, 1) — coordinata di valutazione (k critico)
        """
        b = self.branch(n_k_sampled)
        t = self.trunk(k_c)
        out = torch.sum(b * t, dim=-1, keepdim=True) + self.bias
        return out