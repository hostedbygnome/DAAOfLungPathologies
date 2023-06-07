import numpy as np
import torch


def to_binary_np(mask: torch.Tensor):
    mask = (mask > 0.5).type(torch.uint8)
    while len(mask.shape) > 2:
        mask = mask.squeeze()
    return mask.detach().numpy().astype(np.uint8)
