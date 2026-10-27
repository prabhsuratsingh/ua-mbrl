"""Dimension-agnostic residual dynamics models and bootstrap ensembles.

Environment-specific concerns are injected through feature_fn,
transition_fn and residual_fn hooks.
"""

from typing import Callable

import numpy as np
import torch
import torch.nn as nn


TensorFeatureFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TensorTransitionFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TensorCostFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
TensorResidualFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def default_features(x, u):
    return torch.cat([x, u], dim=1)


def default_transition(x, delta):
    return x + delta


def default_cost(x, u, target):
    state_error = x - target
    return state_error.square().sum(dim=1) + 0.01 * u.square().sum(dim=1)


def default_residual(x_next, x):
    return x_next - x


class GenericDynamicsModel(nn.Module):
    """Single learned residual dynamics model for arbitrary dimensions."""

    def __init__(
        self,
        state_dim,
        action_dim,
        feature_fn=default_features,
        transition_fn=default_transition,
        residual_fn=default_residual,
        input_normalizer=None,
        hidden_dim=128,
        device=None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_fn = feature_fn
        self.transition_fn = transition_fn
        self.residual_fn = residual_fn
        self.input_normalizer = input_normalizer
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        with torch.no_grad():
            feature_dim = feature_fn(
                torch.zeros(1, state_dim),
                torch.zeros(1, action_dim),
            ).shape[1]
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def predict_delta(self, x, u):
        if self.input_normalizer is not None:
            x, u = self.input_normalizer(x, u)
        return self.net(self.feature_fn(x, u))

    def forward(self, x, u):
        return self.transition_fn(x, self.predict_delta(x, u))

    def train_model(self, data, epochs=100, bootstrap_indices=None, verbose=False):
        x = torch.as_tensor(data["x"], dtype=torch.float32, device=self.device)
        u = torch.as_tensor(data["u"], dtype=torch.float32, device=self.device)
        x_next = torch.as_tensor(data["x_next"], dtype=torch.float32, device=self.device)
        if bootstrap_indices is not None:
            x = x[bootstrap_indices]
            u = u[bootstrap_indices]
            x_next = x_next[bootstrap_indices]
        target = self.residual_fn(x_next, x)
        self.train()
        for epoch in range(epochs):
            prediction = self.predict_delta(x, u)
            loss = self.loss_fn(prediction, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if verbose and epoch % 10 == 0:
                print(f"epoch={epoch:4d} loss={loss.item():.6f}")

    def model_step(self, x, u):
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(1, self.state_dim)
        u_tensor = torch.as_tensor(u, dtype=torch.float32, device=self.device).reshape(1, self.action_dim)
        with torch.no_grad():
            prediction = self(x_tensor, u_tensor)
        return prediction.squeeze(0).cpu().numpy()


class GenericEnsembleDynamics:
    """Bootstrap ensemble with arbitrary state/action dimensions."""

    def __init__(
        self,
        state_dim,
        action_dim,
        ensemble_size=5,
        feature_fn=default_features,
        transition_fn=default_transition,
        residual_fn=default_residual,
        input_normalizer=None,
        hidden_dim=128,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.ensemble_size = ensemble_size
        self._stacked_parameters = None
        self.use_vectorized_inference = True
        self.use_vectorized_training = True
        self.models = [
            GenericDynamicsModel(
                state_dim,
                action_dim,
                feature_fn=feature_fn,
                transition_fn=transition_fn,
                residual_fn=residual_fn,
                input_normalizer=input_normalizer,
                hidden_dim=hidden_dim,
                device=self.device,
            )
            for _ in range(ensemble_size)
        ]

    def forward_all(self, x, u):
        deltas = self._vectorized_deltas(x, u)
        return torch.stack(
            [self.models[0].transition_fn(x, delta) for delta in deltas],
            dim=0,
        )

    def _vectorized_deltas(self, x, u):
        """Evaluate all MLP members with batched matrix multiplications."""
        model = self.models[0]
        if not self.use_vectorized_inference:
            return torch.stack(
                [member.predict_delta(x, u) for member in self.models],
                dim=0,
            )
        if model.input_normalizer is not None:
            x, u = model.input_normalizer(x, u)
        features = model.feature_fn(x, u)
        if self._stacked_parameters is None:
            self.refresh_vectorized_parameters()
        weights, biases = self._stacked_parameters
        hidden = torch.einsum("bf,ehf->ebh", features, weights[0]) + biases[0][:, None]
        hidden = torch.relu(hidden)
        hidden = torch.einsum("ebh,edh->ebd", hidden, weights[1]) + biases[1][:, None]
        hidden = torch.relu(hidden)
        return torch.einsum("ebh,edh->ebd", hidden, weights[2]) + biases[2][:, None]

    def refresh_vectorized_parameters(self):
        """Refresh inference weights after model parameters change."""
        self._stacked_parameters = (
            [
                torch.stack([member.net[index].weight for member in self.models])
                for index in (0, 2, 4)
            ],
            [
                torch.stack([member.net[index].bias for member in self.models])
                for index in (0, 2, 4)
            ],
        )

    def forward_mean_var(self, x, u):
        deltas = self._vectorized_deltas(x, u)
        mean_delta = deltas.mean(dim=0)
        variance = deltas.var(
            dim=0,
            unbiased=deltas.shape[0] > 1,
        )
        mean = self.models[0].transition_fn(x, mean_delta)
        return mean, variance, torch.sqrt(variance)

    def predict_uncertainty(self, x, u):
        _, variance, _ = self.forward_mean_var(x, u)
        return variance.sum(dim=1)

    def model_step(self, x, u):
        """NumPy single-step interface: mean prediction only."""
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(1, self.state_dim)
        u_tensor = torch.as_tensor(u, dtype=torch.float32, device=self.device).reshape(1, self.action_dim)
        with torch.no_grad():
            mean, _, _ = self.forward_mean_var(x_tensor, u_tensor)
        return mean.squeeze(0).cpu().numpy()

    def model_step_with_uncertainty(self, x, u):
        """NumPy single-step interface: mean prediction plus scalar uncertainty."""
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(1, self.state_dim)
        u_tensor = torch.as_tensor(u, dtype=torch.float32, device=self.device).reshape(1, self.action_dim)
        with torch.no_grad():
            mean, variance, _ = self.forward_mean_var(x_tensor, u_tensor)
            uncertainty = variance.sum(dim=1)
        return mean.squeeze(0).cpu().numpy(), uncertainty.squeeze(0).cpu().item()

    def train_ensemble(self, data, epochs=100, seed=0, verbose=False):
        if self.use_vectorized_training:
            self._train_ensemble_vectorized(data, epochs=epochs, seed=seed, verbose=verbose)
        else:
            self._train_ensemble_loop(data, epochs=epochs, seed=seed, verbose=verbose)
        self.refresh_vectorized_parameters()

    def _train_ensemble_loop(self, data, epochs=100, seed=0, verbose=False):
        """Sequential per-member training (E separate forward/backward passes).
        Kept for the same kind of loop-vs-vectorized ablation used for inference."""
        rng = np.random.default_rng(seed)
        sample_count = len(data["x"])
        for model in self.models:
            indices = torch.as_tensor(
                rng.integers(sample_count, size=sample_count),
                dtype=torch.long,
                device=self.device,
            )
            model.train_model(
                data,
                epochs=epochs,
                bootstrap_indices=indices,
                verbose=verbose,
            )

    def _train_ensemble_vectorized(self, data, epochs=100, seed=0, verbose=False):
        """Train all E members with one batched forward/backward per epoch.

        Each member keeps its own bootstrap-resampled data and its own real
        nn.Parameter/optimizer (so refresh_vectorized_parameters and per-member
        state stay valid); only the E-fold forward/backward computation is
        batched via stacked-parameter einsum, matching the technique already
        used for inference in _vectorized_deltas.
        """
        rng = np.random.default_rng(seed)
        sample_count = len(data["x"])
        x_all = torch.as_tensor(data["x"], dtype=torch.float32, device=self.device)
        u_all = torch.as_tensor(data["u"], dtype=torch.float32, device=self.device)
        x_next_all = torch.as_tensor(data["x_next"], dtype=torch.float32, device=self.device)

        indices = torch.as_tensor(
            np.stack(
                [rng.integers(sample_count, size=sample_count) for _ in range(self.ensemble_size)]
            ),
            dtype=torch.long,
            device=self.device,
        )  # [E, N]
        x_members = x_all[indices]
        u_members = u_all[indices]
        x_next_members = x_next_all[indices]

        base = self.models[0]
        flat_x = x_members.reshape(-1, self.state_dim)
        flat_u = u_members.reshape(-1, self.action_dim)
        flat_x_next = x_next_members.reshape(-1, self.state_dim)

        # feature_fn/residual_fn/input_normalizer are static (parameter-free)
        # transforms of the data, so -- unlike the network weights -- they
        # only need computing once, not once per epoch.
        target = base.residual_fn(flat_x_next, flat_x).reshape(
            self.ensemble_size, sample_count, self.state_dim
        )
        if base.input_normalizer is not None:
            norm_x, norm_u = base.input_normalizer(flat_x, flat_u)
        else:
            norm_x, norm_u = flat_x, flat_u
        features = base.feature_fn(norm_x, norm_u).reshape(self.ensemble_size, sample_count, -1)

        for model in self.models:
            model.train()

        for epoch in range(epochs):
            weights = [torch.stack([m.net[index].weight for m in self.models]) for index in (0, 2, 4)]
            biases = [torch.stack([m.net[index].bias for m in self.models]) for index in (0, 2, 4)]

            hidden = torch.einsum("ebf,ehf->ebh", features, weights[0]) + biases[0][:, None, :]
            hidden = torch.relu(hidden)
            hidden = torch.einsum("ebh,edh->ebd", hidden, weights[1]) + biases[1][:, None, :]
            hidden = torch.relu(hidden)
            predictions = torch.einsum("ebh,edh->ebd", hidden, weights[2]) + biases[2][:, None, :]

            per_member_loss = ((predictions - target) ** 2).mean(dim=(1, 2))
            total_loss = per_member_loss.sum()

            for model in self.models:
                model.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            for model in self.models:
                model.optimizer.step()

            if verbose and epoch % 10 == 0:
                print(f"epoch={epoch:4d} mean_loss={per_member_loss.mean().item():.6f}")
