"""Input normalization for learned dynamics models."""

import torch


class AffineInputNormalizer:
    """Normalize state/action inputs while preserving physical outputs."""

    def __init__(self, state_mean, state_std, action_mean, action_std, device):
        self.state_mean = torch.as_tensor(state_mean, dtype=torch.float32, device=device)
        self.state_std = torch.as_tensor(state_std, dtype=torch.float32, device=device).clamp_min(1e-6)
        self.action_mean = torch.as_tensor(action_mean, dtype=torch.float32, device=device)
        self.action_std = torch.as_tensor(action_std, dtype=torch.float32, device=device).clamp_min(1e-6)

    def __call__(self, x, u):
        normalized_x = (x - self.state_mean) / self.state_std
        normalized_u = (u - self.action_mean) / self.action_std
        return normalized_x, normalized_u
