"""Adaptive MPPI with a second, severe-OOD threshold that overrides the plan
with a model-free velocity-damping action."""

import numpy as np
import torch

from ua_mbrl.core.mppi import AdaptiveGenericMPPI


class FallbackAdaptiveMPPI(AdaptiveGenericMPPI):
    """AdaptiveGenericMPPI plus a second, higher threshold that overrides the
    chosen action with a model-free damping law instead of trusting the
    ensemble's plan.

    The damping law is u = clip(-gain * x[velocity_index], low, high); the
    pendulum uses velocity_index=1 and bounds (-5, 5).
    """

    def __init__(
        self,
        *args,
        severe_threshold=None,
        fallback_gain=2.0,
        fallback_bounds=(-5.0, 5.0),
        velocity_index=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.severe_threshold = severe_threshold
        self.fallback_gain = fallback_gain
        self.fallback_bounds = fallback_bounds
        self.velocity_index = velocity_index
        self.fallback_triggered_last = False

    @torch.no_grad()
    def command(self, x0):
        action = super().command(x0)
        if self.severe_threshold is not None and self.last_pilot_uncertainty >= self.severe_threshold:
            self.fallback_triggered_last = True
            state = np.asarray(x0, dtype=np.float32).reshape(-1)
            velocity = float(state[self.velocity_index])
            low, high = self.fallback_bounds
            fallback_u = float(np.clip(-self.fallback_gain * velocity, low, high))
            return np.asarray([fallback_u], dtype=np.float32)
        self.fallback_triggered_last = False
        return action
