"""GPU-vectorized MPPI controllers: fixed-K, uncertainty-gated adaptive-K,
and seed-batched fixed-K."""

import numpy as np
import torch

from ua_mbrl.core.dynamics import default_cost

# Process-wide default for AdaptiveGenericMPPI(pilot_rng=None); set by
# `ua-mbrl run --pilot-rng`. See AdaptiveGenericMPPI.__init__.
DEFAULT_PILOT_RNG = "isolated"


def set_default_pilot_rng(mode):
    global DEFAULT_PILOT_RNG
    if mode not in ("isolated", "shared"):
        raise ValueError(f"pilot_rng must be 'isolated' or 'shared', not {mode!r}")
    DEFAULT_PILOT_RNG = mode


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
        pilot_rng=None,
        **kwargs,
    ):
        super().__init__(*args, rollouts=max_rollouts, **kwargs)
        pilot_rng = pilot_rng or DEFAULT_PILOT_RNG
        if pilot_rng not in ("isolated", "shared"):
            raise ValueError(f"pilot_rng must be 'isolated' or 'shared', not {pilot_rng!r}")
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
        #
        # pilot_rng="shared" reproduces the controller as it was before that
        # fix (pilot noise drawn from the global stream, no private generator).
        # Results produced before the fix -- tab:confound, tab:mujoco, the
        # reacher table, the narrow-ensemble fallback rate -- need it to
        # reproduce exactly. Do not use it for new experiments.
        self.pilot_rng = pilot_rng
        self.pilot_generator = torch.Generator(device=self.device) if pilot_rng == "isolated" else None
        self._reseed_pilot_generator()

    def _reseed_pilot_generator(self):
        if self.pilot_generator is None:
            return
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
