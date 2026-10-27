"""Device selection and GPU timing helpers."""

import torch


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = default_device()


def synchronize(device=None):
    """torch.cuda.synchronize() when running on CUDA; a no-op on CPU."""
    device = torch.device(device) if device is not None else DEVICE
    if device.type == "cuda":
        torch.cuda.synchronize(device)
