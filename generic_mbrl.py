"""Dimension-agnostic ensemble dynamics and MPPI primitives.

Environment-specific concerns are injected through:
- feature_fn: state/action -> model features
- transition_fn: state/delta -> next state
- cost_fn: state/action/target -> per-step task cost

This keeps the controller independent of pendulum or MuJoCo details.
"""

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


TensorFeatureFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TensorTransitionFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TensorCostFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
TensorResidualFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


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


class GenericMPPI:
    """GPU-vectorized MPPI for arbitrary state/action dimensions."""

    def __init__(
        self,
        dynamics,
        action_dim,
        horizon,
        rollouts,
        action_low,
        action_high,
        target,
        cost_fn=default_cost,
        dt=1.0,
        temperature=1.0,
        noise_sigma=1.0,
        use_uncertainty=False,
        uncertainty_weight=0.0,
        device=None,
    ):
        self.dynamics = dynamics
        self.action_dim = action_dim
        self.horizon = horizon
        self.rollouts = rollouts
        self.device = torch.device(device or dynamics.device)
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.target = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        self.cost_fn = cost_fn
        self.dt = dt
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.use_uncertainty = use_uncertainty
        self.uncertainty_weight = uncertainty_weight
        self.nominal = torch.zeros(horizon, action_dim, device=self.device)
        self.last_ess = None

    def reset(self):
        self.nominal.zero_()

    @torch.no_grad()
    def command(self, x0):
        x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device).reshape(-1)
        noise = self.noise_sigma * torch.randn(
            self.rollouts, self.horizon, self.action_dim, device=self.device
        )
        actions = torch.empty_like(noise)
        costs = torch.zeros(self.rollouts, device=self.device)
        x = x0.expand(self.rollouts, -1).clone()

        for step in range(self.horizon):
            nominal = self.nominal[step]
            action = torch.clamp(nominal + noise[:, step], self.action_low, self.action_high)
            actions[:, step] = action - nominal
            if hasattr(self.dynamics, "forward_mean_var"):
                x, variance, _ = self.dynamics.forward_mean_var(x, action)
                uncertainty = variance.sum(dim=1)
            else:
                x = self.dynamics(x, action)
                uncertainty = None
            costs += self.cost_fn(x, action, self.target) * self.dt
            if self.use_uncertainty and uncertainty is not None:
                costs += self.uncertainty_weight * uncertainty * self.dt

        costs += self.cost_fn(x, torch.zeros_like(actions[:, 0]), self.target)
        weights = torch.softmax(-(costs - costs.min()) / self.temperature, dim=0)
        # Effective sample size of the importance weights: how many of the K
        # rollouts are actually contributing to the weighted average, a
        # direct measure of estimator quality/degeneracy (not just K itself).
        self.last_ess = float((1.0 / torch.sum(weights ** 2)).item())
        self.nominal += torch.sum(weights[:, None, None] * actions, dim=0)
        self.nominal.clamp_(self.action_low, self.action_high)
        action = self.nominal[0].clone()
        self.nominal[:-1] = self.nominal[1:].clone()
        self.nominal[-1].zero_()
        return action.cpu().numpy()


class AdaptiveGenericMPPI(GenericMPPI):
    """Generic MPPI that allocates rollouts from pilot trajectory uncertainty."""

    def __init__(
        self,
        *args,
        min_rollouts,
        max_rollouts,
        uncertainty_threshold,
        pilot_rollouts=32,
        pilot_horizon=5,
        min_noise_sigma=None,
        max_noise_sigma=None,
        **kwargs,
    ):
        super().__init__(*args, rollouts=max_rollouts, **kwargs)
        self.min_rollouts = min_rollouts
        self.max_rollouts = max_rollouts
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.pilot_rollouts = pilot_rollouts
        self.pilot_horizon = min(pilot_horizon, self.horizon)
        self.last_rollouts = max_rollouts
        self.last_pilot_uncertainty = 0.0
        # Exploration noise scale must be matched to the rollout budget
        # actually in use (a fixed noise scale across a K sweep was found to
        # be a severe, K-growing under-exploration handicap -- see
        # research_status.tex's "more compute hurts" finding and its
        # resolution). The pilot keeps its own fixed scale so the calibrated
        # threshold's semantics don't shift when min/max noise differ.
        self.pilot_noise_sigma = self.noise_sigma
        self.min_noise_sigma = min_noise_sigma if min_noise_sigma is not None else self.noise_sigma
        self.max_noise_sigma = max_noise_sigma if max_noise_sigma is not None else self.noise_sigma
        # The pilot must not perturb the main rollout's noise draws: on this
        # task, even functionally-inert extra RNG consumption (same K, same
        # sigma, no escalation) was found to swing closed-loop success by
        # tens of points purely by shifting the global stream. Giving the
        # pilot its own generator, seeded once (deterministically, from
        # whatever the global seed was at reset time) rather than drawn
        # per-step from the shared stream, isolates the two.
        self.pilot_generator = torch.Generator(device=self.device)
        self._reseed_pilot_generator()

    def _reseed_pilot_generator(self):
        pilot_seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        self.pilot_generator.manual_seed(pilot_seed)

    def reset(self):
        super().reset()
        self._reseed_pilot_generator()

    @torch.no_grad()
    def pilot_uncertainty(self, x0):
        x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device).reshape(-1)
        x = x0.expand(self.pilot_rollouts, -1).clone()
        noise = self.pilot_noise_sigma * torch.randn(
            self.pilot_rollouts,
            self.pilot_horizon,
            self.action_dim,
            device=self.device,
            generator=self.pilot_generator,
        )
        uncertainty = torch.zeros(self.pilot_rollouts, device=self.device)
        for step in range(self.pilot_horizon):
            action = torch.clamp(
                self.nominal[step] + noise[:, step],
                self.action_low,
                self.action_high,
            )
            x, variance, _ = self.dynamics.forward_mean_var(x, action)
            uncertainty += variance.sum(dim=1)
        return torch.mean(uncertainty / self.pilot_horizon).item()

    @torch.no_grad()
    def calibrate_threshold(self, states, percentile=95.0):
        values = [self.pilot_uncertainty(state) for state in states]
        self.uncertainty_threshold = float(np.percentile(values, percentile))
        return self.uncertainty_threshold

    @torch.no_grad()
    def command(self, x0):
        self.last_pilot_uncertainty = self.pilot_uncertainty(x0)
        if self.last_pilot_uncertainty >= self.uncertainty_threshold:
            self.last_rollouts = self.max_rollouts
            self.noise_sigma = self.max_noise_sigma
        else:
            self.last_rollouts = self.min_rollouts
            self.noise_sigma = self.min_noise_sigma
        self.rollouts = self.last_rollouts
        return super().command(x0)


class SeedBatchedGenericMPPI:
    """Fixed-rollout MPPI batched over an extra independent-seed dimension S.

    Runs S independent MPPI instances (different initial states, independent
    noise draws) that share ONE dynamics model, as a single batched tensor
    program instead of a Python loop over seeds. Rollouts (K) and seeds (S)
    are both flattened into the dynamics model's batch dimension for the
    per-step model call; the MPPI softmax weighting and nominal-control
    update stay strictly per-seed (reshaped back to [S, K, ...] before that
    step) so seeds cannot influence each other's control update -- this is
    the correctness constraint that makes "seed" a valid extra batch axis
    alongside rollouts and ensemble members.

    Only fixed-K rollouts are supported here. Adaptive per-seed rollout
    counts would need either padding-with-masking (which spends the same
    compute as fixed-max-K per seed, defeating adaptive-K's purpose) or
    partitioning seeds by decided K into separate batched calls; neither is
    implemented yet, so use per-seed AdaptiveGenericMPPI in a loop for that.
    """

    def __init__(
        self,
        dynamics,
        num_seeds,
        action_dim,
        horizon,
        rollouts,
        action_low,
        action_high,
        target,
        cost_fn=default_cost,
        dt=1.0,
        temperature=1.0,
        noise_sigma=1.0,
        device=None,
    ):
        self.dynamics = dynamics
        self.S = num_seeds
        self.action_dim = action_dim
        self.horizon = horizon
        self.rollouts = rollouts
        self.device = torch.device(device or dynamics.device)
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.target = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        self.cost_fn = cost_fn
        self.dt = dt
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.nominal = torch.zeros(self.S, horizon, action_dim, device=self.device)

    def reset(self):
        self.nominal.zero_()

    @torch.no_grad()
    def command(self, x0):
        """x0: [S, state_dim] (one initial state per seed). Returns [S, action_dim]."""
        x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device).reshape(self.S, -1)
        state_dim = x0.shape[1]
        noise = self.noise_sigma * torch.randn(
            self.S, self.rollouts, self.horizon, self.action_dim, device=self.device
        )
        actions = torch.empty_like(noise)
        costs = torch.zeros(self.S, self.rollouts, device=self.device)
        x = (
            x0.unsqueeze(1)
            .expand(self.S, self.rollouts, state_dim)
            .reshape(self.S * self.rollouts, state_dim)
            .clone()
        )

        for step in range(self.horizon):
            nominal = self.nominal[:, step]  # [S, action_dim]
            action = torch.clamp(
                nominal.unsqueeze(1) + noise[:, :, step],
                self.action_low,
                self.action_high,
            )  # [S, K, action_dim]
            actions[:, :, step] = action - nominal.unsqueeze(1)
            action_flat = action.reshape(self.S * self.rollouts, self.action_dim)

            if hasattr(self.dynamics, "forward_mean_var"):
                x, _, _ = self.dynamics.forward_mean_var(x, action_flat)
            else:
                x = self.dynamics(x, action_flat)

            step_cost = self.cost_fn(x, action_flat, self.target).reshape(self.S, self.rollouts)
            costs += step_cost * self.dt

        terminal_action = torch.zeros_like(action_flat)
        costs += self.cost_fn(x, terminal_action, self.target).reshape(self.S, self.rollouts)

        min_costs = costs.min(dim=1, keepdim=True).values
        weights = torch.softmax(-(costs - min_costs) / self.temperature, dim=1)  # [S, K]
        self.nominal += torch.sum(weights[:, :, None, None] * actions, dim=1)  # [S, H, A]
        self.nominal.clamp_(self.action_low, self.action_high)
        action_out = self.nominal[:, 0].clone()  # [S, action_dim]
        self.nominal[:, :-1] = self.nominal[:, 1:].clone()
        self.nominal[:, -1] = 0.0
        return action_out.cpu().numpy()
