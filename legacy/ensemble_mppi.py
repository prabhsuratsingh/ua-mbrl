"""
Ensemble-based MPPI Controller
-------------------------------

GPU-vectorized MPPI with support for:
- Single model MPPI (baseline)
- Ensemble mean prediction MPPI
- Uncertainty-aware MPPI (uncertainty in cost)
"""

import numpy as np
import torch
from typing import Optional, Tuple


class EnsembleMPPIController:
    """
    GPU-vectorized MPPI with ensemble support.
    
    The horizon dimension N is sequential (dynamics are recurrent).
    The rollout dimension K is fully vectorized.
    
    Supports three modes:
    1. use_uncertainty=False: Standard MPPI with ensemble mean
    2. use_uncertainty=True: Cost includes uncertainty term
    3. adaptive_K=True: Adjust K based on uncertainty (Phase 5)
    """

    def __init__(
        self,
        dynamics_model,
        horizon=30,
        rollouts=1000,
        dt=0.1,
        lam=1.0,
        noise_sigma=1.0,
        x_target=np.array([np.pi, 0.0]),
        u_weight=0.01,
        u_min=-5.0,
        u_max=5.0,
        use_uncertainty=False,
        uncertainty_weight=0.0,
        device=None,
    ):
        """
        Args:
            dynamics_model: EnsembleDynamicsModel or DynamicsModel
            horizon: N (planning horizon)
            rollouts: K (number of trajectory samples)
            dt: time step
            lam: MPPI temperature parameter
            noise_sigma: exploration noise scale
            x_target: goal state
            u_weight: control regularization weight
            u_min, u_max: action bounds
            use_uncertainty: whether to include uncertainty in cost
            uncertainty_weight: β in J = J_task + β * J_uncertainty
            device: torch device
        """
        self.model = dynamics_model

        if device is None:
            if hasattr(dynamics_model, "device"):
                self.device = dynamics_model.device
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            self.device = torch.device(device)

        self.N = horizon
        self.K = rollouts
        self.dt = dt

        self.lam = lam
        self.noise_sigma = noise_sigma

        self.x_target = torch.as_tensor(
            x_target,
            dtype=torch.float32,
            device=self.device,
        )

        self.u_weight = u_weight
        self.u_min = u_min
        self.u_max = u_max

        self.use_uncertainty = use_uncertainty
        self.uncertainty_weight = uncertainty_weight

        # Nominal control sequence
        self.U = torch.zeros(
            self.N,
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self):
        """Reset nominal control sequence."""
        self.U.zero_()

    @torch.no_grad()
    def command(
        self,
        x0,
        diagnose=False,
    ) -> Tuple[float, Optional[dict]]:
        """
        One vectorized MPPI update.

        Args:
            x0: Initial state [2] numpy or torch
            diagnose: Return diagnostic info

        Returns:
            u0: Control action (scalar)
            diagnostics: (optional) dict with diagnostic info
        """
        x0 = torch.as_tensor(
            x0,
            dtype=torch.float32,
            device=self.device,
        ).reshape(2)

        # Sample perturbations on GPU
        delta_u = (
            self.noise_sigma
            * torch.randn(
                self.K,
                self.N,
                device=self.device,
            )
        )

        actual_delta_u = torch.empty_like(delta_u)
        costs = torch.zeros(
            self.K,
            dtype=torch.float32,
            device=self.device,
        )

        diagnostics = {
            "imagined_errors": [],
            "imagined_uncertainties": [],
        } if diagnose else None

        # Expand initial state to K rollouts
        x = x0.unsqueeze(0).expand(self.K, -1).clone()

        # Sequential loop over horizon
        for i in range(self.N):

            u_nominal = self.U[i]
            u_sampled = u_nominal + delta_u[:, i]
            u = torch.clamp(u_sampled, self.u_min, self.u_max)

            actual_delta_u[:, i] = u - u_nominal

            # ------------------------------------------------
            # Dynamics prediction
            # ------------------------------------------------
            # Check if ensemble or single model
            if hasattr(self.model, 'forward_mean_var'):
                # Ensemble model
                x_next, uncertainty = self._ensemble_step(
                    x, u.unsqueeze(1)
                )
            else:
                # Single model
                x_next = self.model(x, u.unsqueeze(1))
                uncertainty = None

            if diagnose and uncertainty is not None:
                diagnostics["imagined_uncertainties"].append(
                    uncertainty.cpu().numpy()
                )

            x = x_next

            # ------------------------------------------------
            # Cost calculation
            # ------------------------------------------------
            theta_error = torch.remainder(
                x[:, 0] - self.x_target[0] + np.pi,
                2 * np.pi,
            ) - np.pi

            omega_error = x[:, 1] - self.x_target[1]

            task_cost = (
                theta_error.square()
                + omega_error.square()
                + self.u_weight * u.square()
            ) * self.dt

            costs += task_cost

            # ------------------------------------------------
            # Optional: uncertainty penalty
            # ------------------------------------------------
            if self.use_uncertainty and uncertainty is not None:
                uncertainty_cost = (
                    self.uncertainty_weight * uncertainty * self.dt
                )
                costs += uncertainty_cost

        # Terminal cost
        theta_error = torch.remainder(
            x[:, 0] - self.x_target[0] + np.pi,
            2 * np.pi,
        ) - np.pi

        omega_error = x[:, 1] - self.x_target[1]

        costs += theta_error.square() + omega_error.square()

        # ------------------------------------------------
        # MPPI weighting
        # ------------------------------------------------
        beta = torch.min(costs)
        weights = torch.exp(-(costs - beta) / self.lam)
        weights /= torch.sum(weights)

        # Update control sequence
        delta_U = weights @ actual_delta_u
        self.U += delta_U
        self.U.clamp_(self.u_min, self.u_max)

        u0 = self.U[0].item()

        # Receding horizon shift
        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1] = 0.0

        if diagnose:
            return u0, diagnostics

        return u0

    def _ensemble_step(self, x, u):
        """
        Ensemble dynamics step.

        x: [K, 2]
        u: [K, 1]

        returns: x_next: [K, 2], uncertainty: [K]
        """
        x_next, variance, _ = self.model.forward_mean_var(x, u)
        uncertainty = torch.sum(variance, dim=1)

        return x_next, uncertainty

    @property
    def uncertainty_enabled(self):
        """Check if uncertainty-aware cost is active."""
        return self.use_uncertainty

    def set_uncertainty_weight(self, beta):
        """Update uncertainty weight (for adaptive experiments)."""
        self.uncertainty_weight = beta
        self.use_uncertainty = (beta > 0)


class AdaptiveEnsembleMPPIController(EnsembleMPPIController):
    """Select a rollout budget from a short ensemble pilot rollout."""

    def __init__(
        self,
        dynamics_model,
        min_rollouts=100,
        max_rollouts=300,
        uncertainty_threshold=0.006,
        pilot_rollouts=25,
        pilot_horizon=5,
        threshold_mode="fixed",
        **kwargs,
    ):
        super().__init__(
            dynamics_model=dynamics_model,
            rollouts=max_rollouts,
            **kwargs,
        )
        self.min_rollouts = min_rollouts
        self.max_rollouts = max_rollouts
        self.uncertainty_threshold = uncertainty_threshold
        self.pilot_rollouts = pilot_rollouts
        self.pilot_horizon = pilot_horizon
        self.threshold_mode = threshold_mode
        self.last_rollouts = max_rollouts
        self.last_pilot_uncertainty = 0.0

    def set_uncertainty_threshold(self, threshold):
        """Set a threshold calibrated from held-out ID uncertainty."""
        self.uncertainty_threshold = float(threshold)

    def _select_rollouts(self, x0):
        if not hasattr(self.model, "forward_mean_var"):
            self.last_rollouts = self.max_rollouts
            return

        self.last_pilot_uncertainty = self._pilot_uncertainty(x0)
        self.last_rollouts = (
            self.max_rollouts
            if self.last_pilot_uncertainty >= self.uncertainty_threshold
            else self.min_rollouts
        )

    def _pilot_uncertainty(self, x0):
        x = torch.as_tensor(
            x0,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 2).expand(self.pilot_rollouts, -1).clone()
        pilot_noise = self.noise_sigma * torch.randn(
            self.pilot_rollouts,
            self.pilot_horizon,
            device=self.device,
        )

        uncertainty_sum = torch.zeros(
            self.pilot_rollouts,
            dtype=torch.float32,
            device=self.device,
        )

        for i in range(self.pilot_horizon):
            u = torch.clamp(
                self.U[i] + pilot_noise[:, i],
                self.u_min,
                self.u_max,
            )
            x, variance, _ = self.model.forward_mean_var(
                x,
                u.unsqueeze(1),
            )
            uncertainty_sum += torch.sum(variance, dim=1)

        return torch.mean(
            uncertainty_sum / self.pilot_horizon
        ).item()

    @torch.no_grad()
    def calibrate_threshold(self, states, percentile=95.0):
        """Calibrate the gate from held-out in-distribution states."""
        uncertainties = [
            self._pilot_uncertainty(state)
            for state in states
        ]
        self.uncertainty_threshold = float(
            np.percentile(uncertainties, percentile)
        )
        return self.uncertainty_threshold

    @torch.no_grad()
    def command(self, x0, diagnose=False):
        x0_array = np.asarray(x0, dtype=np.float32)
        self._select_rollouts(x0_array)
        self.K = self.last_rollouts
        return super().command(x0_array, diagnose=diagnose)


class VanillaMPPIController:
    """
    Standard MPPI controller.
    
    Same as in the original GPU implementation.
    Used as baseline for comparison.
    """

    def __init__(
        self,
        dynamics_model,
        horizon=30,
        rollouts=1000,
        dt=0.1,
        lam=1.0,
        noise_sigma=1.0,
        x_target=np.array([np.pi, 0.0]),
        u_weight=0.01,
        u_min=-5.0,
        u_max=5.0,
        device=None,
    ):
        self.model = dynamics_model

        if device is None:
            if hasattr(dynamics_model, "device"):
                self.device = dynamics_model.device
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            self.device = torch.device(device)

        self.N = horizon
        self.K = rollouts
        self.dt = dt
        self.lam = lam
        self.noise_sigma = noise_sigma

        self.x_target = torch.as_tensor(
            x_target,
            dtype=torch.float32,
            device=self.device,
        )

        self.u_weight = u_weight
        self.u_min = u_min
        self.u_max = u_max

        self.U = torch.zeros(
            self.N,
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self):
        self.U.zero_()

    @torch.no_grad()
    def command(self, x0):
        """Standard MPPI command."""
        x0 = torch.as_tensor(
            x0,
            dtype=torch.float32,
            device=self.device,
        ).reshape(2)

        delta_u = (
            self.noise_sigma
            * torch.randn(
                self.K,
                self.N,
                device=self.device,
            )
        )

        actual_delta_u = torch.empty_like(delta_u)
        costs = torch.zeros(
            self.K,
            dtype=torch.float32,
            device=self.device,
        )

        x = x0.unsqueeze(0).expand(self.K, -1).clone()

        for i in range(self.N):
            u_nominal = self.U[i]
            u_sampled = u_nominal + delta_u[:, i]
            u = torch.clamp(u_sampled, self.u_min, self.u_max)

            actual_delta_u[:, i] = u - u_nominal

            x = self.model(x, u.unsqueeze(1))

            theta_error = torch.remainder(
                x[:, 0] - self.x_target[0] + np.pi,
                2 * np.pi,
            ) - np.pi

            omega_error = x[:, 1] - self.x_target[1]

            costs += (
                theta_error.square()
                + omega_error.square()
                + self.u_weight * u.square()
            ) * self.dt

        # Terminal cost
        theta_error = torch.remainder(
            x[:, 0] - self.x_target[0] + np.pi,
            2 * np.pi,
        ) - np.pi

        omega_error = x[:, 1] - self.x_target[1]

        costs += theta_error.square() + omega_error.square()

        # MPPI weights
        beta = torch.min(costs)
        weights = torch.exp(-(costs - beta) / self.lam)
        weights /= torch.sum(weights)

        delta_U = weights @ actual_delta_u
        self.U += delta_U
        self.U.clamp_(self.u_min, self.u_max)

        u0 = self.U[0].item()

        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1] = 0.0

        return u0
