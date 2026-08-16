import torch

def initial_condition_log(k, A=1.0, alpha=5/3):
    """Versione in log-spazio: log(n_k) = log(A) - alpha*log(k)"""
    return torch.log(torch.tensor(A)) - alpha * torch.log(k)