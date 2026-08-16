import torch

DEVICE = torch.device("cpu")
LR = 1e-2
EPOCHS = 7000
N_COLLOCATION = 5000
N_BOUNDARY = 500

T_MAX = 1.0
K_MIN = 0.1
K_MAX = 1.0
C = 1.0  # velocità di trasporto
