"""
Probabilistic Ensemble Dynamics Model
--------------------------------------

Trains an ensemble of E neural network dynamics models.
Computes mean and uncertainty (variance) predictions.

Used for uncertainty-aware MBRL research.
"""

import numpy as np
import torch
import torch.nn as nn

# GPU is used for training and batched predictions.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicsModel(nn.Module):
    """
    Single deterministic dynamics model.
    
    Input: [sin(theta), cos(theta), omega, u]
    Output: [delta_theta, delta_omega]
    """

    def __init__(self, device=DEVICE):
        super().__init__()

        self.device = torch.device(device)

        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        self.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
        )
        self.loss_fn = nn.MSELoss()

    def predict(self, x, u):
        """
        Predict state deltas.
        
        x: [B, 2]
        u: [B, 1]
        returns: [B, 2]
        """
        theta = x[:, 0:1]
        omega = x[:, 1:2]

        model_input = torch.cat(
            [
                torch.sin(theta),
                torch.cos(theta),
                omega,
                u,
            ],
            dim=1,
        )

        return self.net(model_input)

    def forward(self, x, u):
        """
        Predict next state.
        
        x: [B, 2]
        u: [B, 1]
        returns: [B, 2]
        """
        delta = self.predict(x, u)

        theta_next = x[:, 0:1] + delta[:, 0:1]
        omega_next = x[:, 1:2] + delta[:, 1:2]

        theta_next = torch.remainder(
            theta_next + np.pi,
            2 * np.pi,
        ) - np.pi

        return torch.cat(
            [theta_next, omega_next],
            dim=1,
        )

    def model_step(self, x, u):
        """NumPy single-step interface."""
        x_tensor = torch.as_tensor(
            x,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 2)

        u_tensor = torch.as_tensor(
            [u],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 1)

        with torch.inference_mode():
            x_next = self.forward(
                x_tensor,
                u_tensor,
            )

        return x_next.squeeze(0).cpu().numpy()

    def train_model(
        self,
        x,
        u,
        x_next,
        epochs=100,
        verbose=True,
    ):
        """
        Train on batch data.
        
        x, u, x_next: torch tensors already on device
        """
        # Periodic target for theta
        delta_theta = torch.remainder(
            x_next[:, 0] - x[:, 0] + np.pi,
            2 * np.pi,
        ) - np.pi

        delta_omega = x_next[:, 1] - x[:, 1]

        target = torch.stack(
            [delta_theta, delta_omega],
            dim=1,
        )

        self.train()

        for epoch in range(epochs):
            prediction = self.predict(x, u)
            loss = self.loss_fn(prediction, target)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch: {epoch:4d} | "
                    f"Loss: {loss.item():.6f}"
                )


class EnsembleDynamicsModel:
    """
    Ensemble of E dynamics models.
    
    Provides:
    - Batched ensemble predictions
    - Mean prediction
    - Uncertainty (epistemic via model variance)
    """

    def __init__(
        self,
        ensemble_size=5,
        device=DEVICE,
    ):
        self.E = ensemble_size
        self.device = torch.device(device)

        # Initialize E independent models
        self.models = [
            DynamicsModel(device=self.device)
            for _ in range(self.E)
        ]

    def forward(self, x, u):
        """
        Compute all E predictions.
        
        x: [B, 2]
        u: [B, 1]
        returns: [E, B, 2]
        """
        predictions = []
        for model in self.models:
            pred = model(x, u)
            predictions.append(pred)
        
        return torch.stack(predictions, dim=0)

    def forward_deltas(self, x, u):
        """Compute unwrapped state-delta predictions: [E, B, 2]."""
        predictions = [model.predict(x, u) for model in self.models]
        return torch.stack(predictions, dim=0)

    def forward_mean_var(self, x, u):
        """
        Compute mean and variance across ensemble.
        
        x: [B, 2]
        u: [B, 1]
        returns: 
            mean: [B, 2]
            var: [B, 2]
            std: [B, 2]
        """
        deltas = self.forward_deltas(x, u)  # [E, B, 2]

        mean_delta = torch.mean(deltas, dim=0)  # [B, 2]
        var = torch.var(deltas, dim=0, unbiased=True)  # [B, 2]
        std = torch.sqrt(var)  # [B, 2]

        mean = x + mean_delta
        mean_theta = torch.remainder(
            mean[:, 0:1] + np.pi,
            2 * np.pi,
        ) - np.pi
        mean = torch.cat([mean_theta, mean[:, 1:2]], dim=1)

        return mean, var, std

    def predict_uncertainty(self, x, u):
        """
        Scalar uncertainty metric.
        
        For each (x, u), compute trace of covariance matrix
        (sum of variances across dimensions).
        
        x: [B, 2]
        u: [B, 1]
        returns: [B]  -- uncertainty per sample
        """
        _, var, _ = self.forward_mean_var(x, u)
        
        # Sum variances across state dimensions
        uncertainty = torch.sum(var, dim=1)  # [B]
        
        return uncertainty

    def model_step(self, x, u):
        """
        NumPy interface: mean prediction only.
        
        Used for control (receding horizon).
        """
        x_tensor = torch.as_tensor(
            x,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 2)

        u_tensor = torch.as_tensor(
            [u],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 1)

        with torch.inference_mode():
            mean, _, _ = self.forward_mean_var(
                x_tensor,
                u_tensor,
            )

        return mean.squeeze(0).cpu().numpy()

    def model_step_with_uncertainty(self, x, u):
        """
        NumPy interface: mean + uncertainty.
        
        x: numpy [2]
        u: scalar
        returns: (x_next, uncertainty)
            x_next: numpy [2]
            uncertainty: scalar
        """
        x_tensor = torch.as_tensor(
            x,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 2)

        u_tensor = torch.as_tensor(
            [u],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 1)

        with torch.inference_mode():
            mean, variance, _ = self.forward_mean_var(
                x_tensor,
                u_tensor,
            )
            uncertainty = torch.sum(variance, dim=1)

        return (
            mean.squeeze(0).cpu().numpy(),
            uncertainty.squeeze(0).cpu().item(),
        )

    def train_ensemble(
        self,
        data,
        epochs=100,
        verbose=True,
        bootstrap=True,
        seed=0,
    ):
        """
        Train all E models on the same dataset.
        
        data: dict with keys "x", "u", "x_next"
        """
        x = torch.as_tensor(
            data["x"],
            dtype=torch.float32,
            device=self.device,
        )
        u = torch.as_tensor(
            data["u"],
            dtype=torch.float32,
            device=self.device,
        )
        x_next = torch.as_tensor(
            data["x_next"],
            dtype=torch.float32,
            device=self.device,
        )

        rng = np.random.default_rng(seed)

        for e, model in enumerate(self.models):
            if verbose:
                print(f"\n--- Training model {e + 1}/{self.E} ---")
            
            if bootstrap:
                indices = rng.integers(
                    0,
                    len(data["x"]),
                    size=len(data["x"]),
                )
                model_x = x[indices]
                model_u = u[indices]
                model_x_next = x_next[indices]
            else:
                model_x = x
                model_u = u
                model_x_next = x_next

            model.train_model(
                model_x,
                model_u,
                model_x_next,
                epochs=epochs,
                verbose=verbose,
            )

    def get_all_predictions(self, x, u):
        """
        Get all E predictions as numpy.
        
        x: [B, 2]
        u: [B, 1]
        returns: [E, B, 2]
        """
        with torch.inference_mode():
            predictions = self.forward(x, u)
        
        return predictions.cpu().numpy()

    def evaluate_ensemble_disagreement(
        self,
        test_data,
    ):
        """
        Diagnostic: measure ensemble disagreement.
        
        For each test sample, compute:
        - Ensemble predictions
        - Mean
        - Std dev
        - Actual error vs true next state
        
        returns: dict with arrays of statistics
        """
        x = torch.as_tensor(
            test_data["x"],
            dtype=torch.float32,
            device=self.device,
        )
        u = torch.as_tensor(
            test_data["u"],
            dtype=torch.float32,
            device=self.device,
        )
        x_next_true = torch.as_tensor(
            test_data["x_next"],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.inference_mode():
            mean, var, std = self.forward_mean_var(x, u)
            uncertainty = self.predict_uncertainty(x, u)

        # Compute error of mean prediction
        error = mean - x_next_true

        # Wrap angle error
        error[:, 0] = torch.remainder(
            error[:, 0] + np.pi,
            2 * np.pi,
        ) - np.pi

        # Absolute error
        abs_error = torch.abs(error)
        
        # Total error magnitude per sample
        error_magnitude = torch.sqrt(
            torch.sum(error ** 2, dim=1)
        )

        return {
            "mean": mean.cpu().numpy(),
            "std": std.cpu().numpy(),
            "uncertainty": uncertainty.cpu().numpy(),
            "error": error.cpu().numpy(),
            "abs_error": abs_error.cpu().numpy(),
            "error_magnitude": error_magnitude.cpu().numpy(),
            "x_true": x_next_true.cpu().numpy(),
        }

    def save(self, path):
        """Save all ensemble models."""
        state_dict = {
            f"model_{e}": self.models[e].state_dict()
            for e in range(self.E)
        }
        torch.save(state_dict, path)

    def load(self, path):
        """Load all ensemble models."""
        state_dict = torch.load(path)
        for e in range(self.E):
            self.models[e].load_state_dict(
                state_dict[f"model_{e}"]
            )


class AnalyticalEnsembleWrapper:
    """
    Wraps analytical dynamics to match ensemble interface.
    
    All E models return the same true prediction (no uncertainty).
    Used as baseline for comparison.
    """

    def __init__(self, device=DEVICE):
        self.device = torch.device(device)
        self.E = 5  # Match ensemble size

    def __call__(self, x, u, analytical_dynamics_fn):
        """
        x: [B, 2]
        u: [B]
        returns: [E, B, 2] with identical predictions
        """
        # Compute true dynamics
        x_next_true = analytical_dynamics_fn(x, u)

        # Replicate across ensemble dimension
        predictions = x_next_true.unsqueeze(0).repeat(self.E, 1, 1)

        return predictions

    def forward_mean_var(self, x, u, analytical_dynamics_fn):
        """
        Returns zero uncertainty.
        """
        x_next_true = analytical_dynamics_fn(x, u)

        mean = x_next_true
        var = torch.zeros_like(x_next_true)
        std = torch.zeros_like(x_next_true)

        return mean, var, std
