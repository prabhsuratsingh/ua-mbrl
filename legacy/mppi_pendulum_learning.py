"""
MPPI Pendulum with Learned Dynamics
-----------------------------------
Compact MBRL experiment:

1. Collect an initial dataset with the analytical pendulum.
2. Train a neural dynamics model.
3. Validate one-step and multi-step prediction.
4. Compare perfect-model MPPI vs learned-model MPPI.
5. Run online MBRL:
       real system -> replay buffer -> retrain model -> repeat

The analytical dynamics are the TRUE environment.
The neural network is used only inside MPPI for imagined rollouts.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

DT = 0.1
G = 9.81
L = 1.0
M = 1.0

U_MIN = -5.0
U_MAX = 5.0

MPPI_HORIZON = 30
MPPI_ROLLOUTS = 1000
MPPI_LAMBDA = 1.0
MPPI_NOISE_SIGMA = 1.0
CONTROL_WEIGHT = 0.01

INITIAL_EPISODES = 20
INITIAL_STEPS = 100

EVAL_EPISODES = 10
EVAL_STEPS = 60

ONLINE_EPISODES = 10
ONLINE_TRAIN_EPOCHS = 50

SEED_DATA = 0
SEED_INITIAL_STATES = 100


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def angle_error(theta, target=np.pi):
    return wrap_angle(theta - target)

def unwrap_angles(theta):
    return np.unwrap(theta)

# ============================================================
# TRUE ENVIRONMENT
# ============================================================

def environment_dynamics(
    x,
    g=G,
    l=L,
    u=0.0,
    m=M,
    dt=DT,
):
    """Analytical pendulum dynamics: this is the true environment."""

    theta, omega = x

    theta_dot = omega
    omega_dot = -(g / l) * np.sin(theta) + u / (m * l**2)

    theta_next = theta + theta_dot * dt
    omega_next = omega + omega_dot * dt

    return np.array(
        [wrap_angle(theta_next), omega_next],
        dtype=np.float32,
    )


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity, state_dim=2, action_dim=1):
        self.capacity = capacity

        self.states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (capacity, action_dim),
            dtype=np.float32,
        )
        self.next_states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )

        self.size = 0
        self.ptr = 0

    def add(self, x, u, x_next):
        self.states[self.ptr] = x
        self.actions[self.ptr] = u
        self.next_states[self.ptr] = x_next

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_all(self):
        return {
            "x": self.states[:self.size],
            "u": self.actions[:self.size],
            "x_next": self.next_states[:self.size],
        }

    def __len__(self):
        return self.size


# ============================================================
# INITIAL DATA COLLECTION
# ============================================================

def collect_initial_data(
    buffer,
    num_episodes=INITIAL_EPISODES,
    steps_per_episode=INITIAL_STEPS,
    action_scale=U_MAX,
    seed=SEED_DATA,
):
    """Collect random transitions from the TRUE environment."""

    np.random.seed(seed)

    for _ in range(num_episodes):
        x = np.array(
            [
                np.random.uniform(-np.pi, np.pi),
                np.random.uniform(-2.0, 2.0),
            ],
            dtype=np.float32,
        )

        for _ in range(steps_per_episode):
            u = np.random.uniform(-action_scale, action_scale)

            x_next = environment_dynamics(x, u=u)

            buffer.add(
                x,
                np.array([u], dtype=np.float32),
                x_next,
            )

            x = x_next


# ============================================================
# LEARNED DYNAMICS MODEL
# ============================================================

class DynamicsModel(nn.Module):
    """
    Learns:

        [sin(theta), cos(theta), omega, u]
                    ->
        [delta_theta, delta_omega]

    Predicting deltas is easier than directly predicting x_next.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        # Keep optimizer state across online retraining.
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
        )
        self.loss_fn = nn.MSELoss()

    def predict(self, x, u):
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
        """NumPy single-step interface used by MPPI."""

        x_tensor = torch.as_tensor(
            x,
            dtype=torch.float32,
        ).reshape(1, 2)

        u_tensor = torch.as_tensor(
            [u],
            dtype=torch.float32,
        ).reshape(1, 1)

        with torch.no_grad():
            x_next = self.forward(
                x_tensor,
                u_tensor,
            )

        return x_next.squeeze(0).numpy()

    def train_dynamics_model(
        self,
        data,
        epochs=100,
        verbose=True,
    ):
        x = torch.as_tensor(
            data["x"],
            dtype=torch.float32,
        )
        u = torch.as_tensor(
            data["u"],
            dtype=torch.float32,
        )
        x_next = torch.as_tensor(
            data["x_next"],
            dtype=torch.float32,
        )

        # Periodic target for theta.
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

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch: {epoch:4d} | "
                    f"Loss: {loss.item():.6f}"
                )


# ============================================================
# COST FUNCTIONS
# ============================================================

def running_cost(
    x,
    x_target=np.array([np.pi, 0.0]),
    u=0.0,
    u_weight=CONTROL_WEIGHT,
):
    theta_error = angle_error(x[0], x_target[0])
    omega_error = x[1] - x_target[1]

    return (
        theta_error**2
        + omega_error**2
        + u_weight * u**2
    )


def terminal_cost(
    x,
    x_target=np.array([np.pi, 0.0]),
):
    theta_error = angle_error(x[0], x_target[0])
    omega_error = x[1] - x_target[1]

    return theta_error**2 + omega_error**2


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_one_step(model, test_data):
    predictions = np.array(
        [
            model.model_step(x, u[0])
            for x, u in zip(
                test_data["x"],
                test_data["u"],
            )
        ]
    )

    errors = predictions - test_data["x_next"]

    # Theta is periodic.
    errors[:, 0] = np.array(
        [
            angle_error(p, t)
            for p, t in zip(
                predictions[:, 0],
                test_data["x_next"][:, 0],
            )
        ]
    )

    print("\n==============================")
    print("ONE-STEP MODEL EVALUATION")
    print("==============================")
    print(f"Theta MSE : {np.mean(errors[:, 0] ** 2):.6f}")
    print(f"Omega MSE : {np.mean(errors[:, 1] ** 2):.6f}")
    print(f"Total MSE : {np.mean(errors ** 2):.6f}")

    return predictions, errors


def evaluate_multistep_model(
    model,
    x0,
    actions,
):
    """Compare recursive learned-model rollout against TRUE dynamics."""

    x_true = np.array(x0, dtype=np.float32)
    x_model = np.array(x0, dtype=np.float32)

    true_trajectory = [x_true.copy()]
    model_trajectory = [x_model.copy()]

    for u in actions:
        x_true = environment_dynamics(
            x_true,
            u=u,
        )

        x_model = model.model_step(
            x_model,
            u,
        )

        true_trajectory.append(x_true.copy())
        model_trajectory.append(x_model.copy())

    true_trajectory = np.array(true_trajectory)
    model_trajectory = np.array(model_trajectory)

    errors = model_trajectory - true_trajectory

    errors[:, 0] = np.array(
        [
            angle_error(p, t)
            for p, t in zip(
                model_trajectory[:, 0],
                true_trajectory[:, 0],
            )
        ]
    )

    return (
        true_trajectory,
        model_trajectory,
        errors,
    )


def evaluate_multistep_summary(
    model,
    test_data,
    steps_per_episode=INITIAL_STEPS,
    num_rollouts=20,
    horizon=30,
    seed=123,
):
    """
    Multi-step validation without crossing episode boundaries.

    The old implementation could accidentally take the last few
    actions of one episode and then continue with the first actions
    of the next episode. This version prevents that.
    """

    rng = np.random.default_rng(seed)

    num_test_episodes = len(test_data["x"]) // steps_per_episode

    if horizon > steps_per_episode:
        raise ValueError(
            "horizon must be <= steps_per_episode"
        )

    theta_rmses = []
    omega_rmses = []
    total_rmses = []

    for _ in range(num_rollouts):
        episode = rng.integers(num_test_episodes)
        start = episode * steps_per_episode

        max_offset = steps_per_episode - horizon
        offset = rng.integers(max_offset + 1)

        start_idx = start + offset

        x0 = test_data["x"][start_idx].copy()
        actions = test_data["u"][
            start_idx:start_idx + horizon,
            0,
        ].copy()

        _, _, errors = evaluate_multistep_model(
            model,
            x0,
            actions,
        )

        theta_rmses.append(
            np.sqrt(np.mean(errors[:, 0] ** 2))
        )
        omega_rmses.append(
            np.sqrt(np.mean(errors[:, 1] ** 2))
        )
        total_rmses.append(
            np.sqrt(np.mean(errors ** 2))
        )

    print("\n==============================")
    print("MULTI-STEP MODEL EVALUATION")
    print("==============================")
    print(f"Rollouts : {num_rollouts}")
    print(f"Horizon  : {horizon}")
    print(
        f"Theta RMSE : "
        f"{np.mean(theta_rmses):.6f} "
        f"+/- {np.std(theta_rmses):.6f}"
    )
    print(
        f"Omega RMSE : "
        f"{np.mean(omega_rmses):.6f} "
        f"+/- {np.std(omega_rmses):.6f}"
    )
    print(
        f"Total RMSE : "
        f"{np.mean(total_rmses):.6f} "
        f"+/- {np.std(total_rmses):.6f}"
    )


# ============================================================
# MPPI
# ============================================================

class MPPIController:
    def __init__(
        self,
        dynamics_model,
        horizon=MPPI_HORIZON,
        rollouts=MPPI_ROLLOUTS,
        dt=DT,
        lam=MPPI_LAMBDA,
        noise_sigma=MPPI_NOISE_SIGMA,
        x_target=np.array([np.pi, 0.0]),
        u_weight=CONTROL_WEIGHT,
        u_min=U_MIN,
        u_max=U_MAX,
    ):
        self.model = dynamics_model

        self.N = horizon
        self.K = rollouts
        self.dt = dt

        self.lam = lam
        self.noise_sigma = noise_sigma

        self.x_target = x_target
        self.u_weight = u_weight

        self.u_min = u_min
        self.u_max = u_max

        # Nominal control sequence.
        self.U = np.zeros(self.N)

    def reset(self):
        self.U[:] = 0.0

    def command(self, x0, diagnose=False):
        """
        One MPPI update.

        All imagined trajectories use self.model.
        Only the returned u0 is applied to the TRUE environment.
        """

        delta_u = (
            self.noise_sigma
            * np.random.randn(self.K, self.N)
        )

        actual_delta_u = np.zeros_like(delta_u)
        costs = np.zeros(self.K)
        imagined_model_errors = []

        for k in range(self.K):
            x = x0.copy()

            for i in range(self.N):
                u_nominal = self.U[i]

                u_sampled = (
                    u_nominal
                    + delta_u[k, i]
                )

                u = np.clip(
                    u_sampled,
                    self.u_min,
                    self.u_max,
                )

                actual_delta_u[k, i] = (
                    u - u_nominal
                )

                # IMPORTANT:
                # learned model is used only for imagination.
                if diagnose:

                    x_model = self.model.model_step(
                        x,
                        u
                    )

                    x_true = environment_dynamics(
                        x,
                        u=u
                    )

                    theta_error = angle_error(
                        x_model[0],
                        x_true[0]
                    )

                    omega_error = (
                        x_model[1]
                        - x_true[1]
                    )

                    imagined_model_errors.append(
                        [
                            theta_error,
                            omega_error
                        ]
                    )

                    x = x_model

                else:

                    x = self.model.model_step(
                        x,
                        u
                    )

                costs[k] += (
                    running_cost(
                        x,
                        self.x_target,
                        u,
                        self.u_weight,
                    )
                    * self.dt
                )

            costs[k] += terminal_cost(
                x,
                self.x_target,
            )

        # Numerically stable importance weights.
        beta = np.min(costs)

        weights = np.exp(
            -(costs - beta) / self.lam
        )
        weights /= np.sum(weights)

        delta_U = weights @ actual_delta_u

        self.U += delta_U
        self.U = np.clip(
            self.U,
            self.u_min,
            self.u_max,
        )

        u0 = self.U[0]

        # Receding-horizon shift.
        self.U = np.roll(self.U, -1)
        self.U[-1] = 0.0

        if diagnose:
            return u0, np.asarray(imagined_model_errors)

        return u0



# ============================================================
# ANALYTICAL MODEL WRAPPER
# ============================================================

class AnalyticalDynamicsModel:
    """Wraps the TRUE analytical dynamics so MPPI can use it."""

    def model_step(self, x, u):
        return environment_dynamics(x, u=u)


# ============================================================
# RUN MPPI ON TRUE ENVIRONMENT
# ============================================================

def run_mppi(
    dynamics_model,
    x0,
    num_steps=EVAL_STEPS,
    seed=0,
    replay_buffer=None,
):
    """
    Run receding-horizon MPPI.

    dynamics_model:
        model used INSIDE MPPI.

    replay_buffer:
        if supplied, real transitions are added to it.
        This is what turns the experiment into online MBRL.
    """

    np.random.seed(seed)

    controller = MPPIController(
        dynamics_model=dynamics_model,
    )

    x = np.array(x0, dtype=np.float32)

    x_history = [x.copy()]
    u_history = []

    model_errors = []

    for _ in range(num_steps):
        # ----------------------------------------------------
        # 1. Plan using the supplied model
        # ----------------------------------------------------
        # u0 = controller.command(x)
        u0, imagined_errors = controller.command(
            x,
            diagnose=True
        )

        # ----------------------------------------------------
        # 2. Ask the model what it predicted
        # ----------------------------------------------------
        x_model_next = dynamics_model.model_step(
            x,
            u0,
        )

        # ----------------------------------------------------
        # 3. Apply action to TRUE environment
        # ----------------------------------------------------
        x_next = environment_dynamics(
            x,
            u=u0,
        )

        # ----------------------------------------------------
        # 4. Store REAL transition
        # ----------------------------------------------------
        if replay_buffer is not None:
            replay_buffer.add(
                x.copy(),
                np.array([u0], dtype=np.float32),
                x_next.copy(),
            )

        # ----------------------------------------------------
        # 5. Measure model error on this real transition
        # ----------------------------------------------------
        error = x_model_next - x_next
        error[0] = angle_error(
            x_model_next[0],
            x_next[0],
        )

        model_errors.append(error)

        x = x_next

        x_history.append(x.copy())
        u_history.append(u0)

    return (
        np.array(x_history),
        np.array(u_history),
        np.array(model_errors),
    )


# ============================================================
# MPPI EVALUATION
# ============================================================

def evaluate_mppi(
    dynamics_model,
    initial_states,
    num_steps=EVAL_STEPS,
    seed=100,
):
    results = []

    for episode, x0 in enumerate(initial_states):
        x_history, u_history, model_errors = run_mppi(
            dynamics_model=dynamics_model,
            x0=x0,
            num_steps=num_steps,
            seed=seed + episode,
        )

        theta_final = x_history[-1, 0]
        omega_final = x_history[-1, 1]

        theta_err = angle_error(theta_final)

        success = (
            abs(theta_err) < 0.1
            and abs(omega_final) < 0.5
        )

        results.append(
            {
                "episode": episode + 1,
                "theta_error": theta_err,
                "final_omega": omega_final,

                "mean_abs_u": np.mean(
                    np.abs(u_history)
                ),

                "max_abs_u": np.max(
                    np.abs(u_history)
                ),

                "model_theta_rmse": np.sqrt(
                    np.mean(model_errors[:, 0] ** 2)
                ),

                "model_omega_rmse": np.sqrt(
                    np.mean(model_errors[:, 1] ** 2)
                ),

                "model_total_rmse": np.sqrt(
                    np.mean(model_errors ** 2)
                ),

                "success": success,

                # NEW
                "trajectory": x_history.copy(),
                "controls": u_history.copy(),
                "model_errors": model_errors.copy(),
            }
        )

        print(
            f"Episode {episode + 1:2d}/{len(initial_states)}: "
            f"x0=({x0[0]: .2f}, {x0[1]: .2f}) | "
            f"theta_err={theta_err: .3f} | "
            f"omega={omega_final: .3f} | "
            f"model_RMSE={results[-1]['model_total_rmse']:.3f} | "
            f"success={success}"
        )

    return results

def inspect_failed_episodes(results):
    failed = [
        r for r in results
        if not r["success"]
    ]

    if not failed:
        print("No failed episodes.")
        return

    print("\n==============================")
    print("FAILED EPISODE ANALYSIS")
    print("==============================")

    for r in failed:

        trajectory = r["trajectory"]
        controls = r["controls"]

        theta = trajectory[:, 0]
        omega = trajectory[:, 1]

        print(
            f"Episode {r['episode']}: "
            f"max |omega| = {np.max(np.abs(omega)):.3f}, "
            f"max |u| = {np.max(np.abs(controls)):.3f}, "
            f"theta range = "
            f"[{theta.min():.3f}, {theta.max():.3f}]"
        )

def summarize_results(results, title):
    successes = np.array(
        [r["success"] for r in results]
    )

    theta_errors = np.array(
        [abs(r["theta_error"]) for r in results]
    )

    omega_errors = np.array(
        [abs(r["final_omega"]) for r in results]
    )

    mean_controls = np.array(
        [r["mean_abs_u"] for r in results]
    )

    max_controls = np.array(
        [r["max_abs_u"] for r in results]
    )

    print("\n==============================")
    print(title)
    print("==============================")

    print(
        f"Success rate       : "
        f"{100 * np.mean(successes):.1f}% "
        f"({np.sum(successes)}/{len(results)})"
    )
    print(
        f"Mean |theta error| : "
        f"{np.mean(theta_errors):.4f}"
    )
    print(
        f"Mean |omega|       : "
        f"{np.mean(omega_errors):.4f}"
    )
    print(
        f"Mean |u|           : "
        f"{np.mean(mean_controls):.4f}"
    )
    print(
        f"Mean max |u|       : "
        f"{np.mean(max_controls):.4f}"
    )
    print(
        f"Overall max |u|    : "
        f"{np.max(max_controls):.4f}"
    )


def compare_model_error(results, label):
    successful = [
        r for r in results
        if r["success"]
    ]

    failed = [
        r for r in results
        if not r["success"]
    ]

    print("\n==============================")
    print(f"MODEL ERROR: {label}")
    print("==============================")

    for name, group in [
        ("Successful", successful),
        ("Failed", failed),
    ]:
        if not group:
            continue

        print(f"{name} episodes ({len(group)}):")

        print(
            f"  Mean total RMSE : "
            f"{np.mean([r['model_total_rmse'] for r in group]):.6f}"
        )
        print(
            f"  Mean theta RMSE : "
            f"{np.mean([r['model_theta_rmse'] for r in group]):.6f}"
        )
        print(
            f"  Mean omega RMSE : "
            f"{np.mean([r['model_omega_rmse'] for r in group]):.6f}"
        )


# ============================================================
# ONLINE MBRL
# ============================================================

def run_online_mbrl(
    initial_data,
    initial_states,
    num_episodes=ONLINE_EPISODES,
    train_epochs=ONLINE_TRAIN_EPOCHS,
):
    """
    Online MBRL loop:

        learned model
              ↓
            MPPI
              ↓
        TRUE environment
              ↓
        replay buffer
              ↓
        retrain model
              ↓
            repeat
    """

    online_buffer = ReplayBuffer(
        capacity=10000,
    )

    # Seed the online buffer with initial random data.
    for x, u, x_next in zip(
        initial_data["x"],
        initial_data["u"],
        initial_data["x_next"],
    ):
        online_buffer.add(x, u, x_next)

    online_model = DynamicsModel()

    print(
        f"\nInitial replay buffer size: "
        f"{len(online_buffer)}"
    )

    # Initial model.
    print("\nInitial model training:")
    online_model.train_dynamics_model(
        online_buffer.get_all(),
        epochs=train_epochs,
    )

    online_results = []

    for episode in range(num_episodes):
        x0 = initial_states[episode]

        print(
            f"\n--- Online episode "
            f"{episode + 1}/{num_episodes} ---"
        )

        x_history, u_history, model_errors = run_mppi(
            dynamics_model=online_model,
            x0=x0,
            num_steps=EVAL_STEPS,
            seed=200 + episode,
            replay_buffer=online_buffer,
        )

        # Retrain on ALL accumulated real data.
        online_model.train_dynamics_model(
            online_buffer.get_all(),
            epochs=train_epochs,
        )

        theta_final = x_history[-1, 0]
        omega_final = x_history[-1, 1]
        theta_err = angle_error(theta_final)

        success = (
            abs(theta_err) < 0.1
            and abs(omega_final) < 0.5
        )

        model_rmse = np.sqrt(
            np.mean(model_errors ** 2)
        )

        result = {
            "episode": episode + 1,
            "theta_error": theta_err,
            "final_omega": omega_final,
            "mean_abs_u": np.mean(
                np.abs(u_history)
            ),
            "max_abs_u": np.max(
                np.abs(u_history)
            ),
            "model_theta_rmse": np.sqrt(
                np.mean(model_errors[:, 0] ** 2)
            ),
            "model_omega_rmse": np.sqrt(
                np.mean(model_errors[:, 1] ** 2)
            ),
            "model_total_rmse": model_rmse,
            "success": success,
            "trajectory": x_history.copy(),
            "controls": u_history.copy(),
            "model_errors": model_errors.copy(),
        }

        online_results.append(result)

        print(
            f"theta error = {theta_err:.3f} | "
            f"omega = {omega_final:.3f} | "
            f"model RMSE = {model_rmse:.3f} | "
            f"success = {success}"
        )
        print(
            f"Replay buffer size: "
            f"{len(online_buffer)}"
        )

    summarize_results(
        online_results,
        "ONLINE MBRL SUMMARY",
    )

    compare_model_error(
        online_results,
        "ONLINE MBRL: SUCCESS vs FAILURE",
    )

    return online_model, online_results


# ============================================================
# PLOTS
# ============================================================

def plot_trajectory(
    x_history,
    u_history,
    filename="runs/learned_mppi_pendulum.png",
):
    dt = DT

    time_x = np.arange(len(x_history)) * dt
    time_u = np.arange(len(u_history)) * dt

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9, 9),
        sharex=True,
    )

    theta_plot = unwrap_angles(x_history[:, 0])

    axes[0].plot(
        time_x,
        theta_plot,
        marker="o",
        markersize=3,
        label=r"$\theta$",
    )
    axes[0].axhline(
        np.pi,
        linestyle="--",
        label=r"target $\theta=\pi$",
    )
    axes[0].set_ylabel(r"$\theta$ [rad]")
    axes[0].set_title("Learned-Model MPPI: Angle")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        time_x,
        x_history[:, 1],
        marker="o",
        markersize=3,
        label=r"$\omega$",
    )
    axes[1].axhline(
        0.0,
        linestyle="--",
        label=r"target $\omega=0$",
    )
    axes[1].set_ylabel(r"$\omega$ [rad/s]")
    axes[1].set_title(
        "Learned-Model MPPI: Angular Velocity"
    )
    axes[1].legend()
    axes[1].grid(True)

    axes[2].step(
        time_u,
        u_history,
        where="post",
        label="applied torque",
    )
    axes[2].set_ylabel("u [Nm]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_title("Learned-Model MPPI: Control")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    os.makedirs("runs", exist_ok=True)

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()


def plot_model_error(
    model_errors,
    filename="runs/on_policy_model_error.png",
):
    time = np.arange(len(model_errors)) * DT

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 6),
        sharex=True,
    )

    axes[0].plot(
        time,
        model_errors[:, 0],
        marker="o",
        markersize=3,
        label="theta prediction error",
    )
    axes[0].axhline(
        0.0,
        linestyle="--",
    )
    axes[0].set_ylabel(
        r"$\Delta\theta$ error [rad]"
    )
    axes[0].set_title(
        "Learned Model: On-Policy Prediction Error"
    )
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        time,
        model_errors[:, 1],
        marker="o",
        markersize=3,
        label="omega prediction error",
    )
    axes[1].axhline(
        0.0,
        linestyle="--",
    )
    axes[1].set_ylabel(
        r"$\Delta\omega$ error [rad/s]"
    )
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    os.makedirs("runs", exist_ok=True)

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Initial random dataset
    # --------------------------------------------------------

    set_seed(SEED_DATA)

    buffer = ReplayBuffer(
        capacity=10000,
    )

    collect_initial_data(
        buffer=buffer,
        num_episodes=INITIAL_EPISODES,
        steps_per_episode=INITIAL_STEPS,
        action_scale=U_MAX,
        seed=SEED_DATA,
    )

    data = buffer.get_all()

    TRAIN_EPISODES = int(0.8 * INITIAL_EPISODES)
    split = TRAIN_EPISODES * INITIAL_STEPS  

    assert split % INITIAL_STEPS == 0

    train_data = {
        "x": data["x"][:split],
        "u": data["u"][:split],
        "x_next": data["x_next"][:split],
    }

    test_data = {
        "x": data["x"][split:],
        "u": data["u"][split:],
        "x_next": data["x_next"][split:],
    }

    print("\nDataset ranges:")
    print(
        f"theta train: "
        f"{train_data['x'][:, 0].min():.3f} "
        f"to {train_data['x'][:, 0].max():.3f}"
    )
    print(
        f"omega train: "
        f"{train_data['x'][:, 1].min():.3f} "
        f"to {train_data['x'][:, 1].max():.3f}"
    )
    print(
        f"theta test:  "
        f"{test_data['x'][:, 0].min():.3f} "
        f"to {test_data['x'][:, 0].max():.3f}"
    )
    print(
        f"omega test:  "
        f"{test_data['x'][:, 1].min():.3f} "
        f"to {test_data['x'][:, 1].max():.3f}"
    )
    print(
        f"action:      "
        f"{data['u'].min():.3f} "
        f"to {data['u'].max():.3f}"
    )
    print(
        f"Number of samples: "
        f"{len(data['x'])}"
    )

    # --------------------------------------------------------
    # 2. Train initial learned dynamics model
    # --------------------------------------------------------

    set_seed(1)

    model = DynamicsModel()

    model.train_dynamics_model(
        train_data,
        epochs=100,
    )

    # --------------------------------------------------------
    # 3. One-step and multi-step validation
    # --------------------------------------------------------

    evaluate_one_step(
        model,
        test_data,
    )

    evaluate_multistep_summary(
        model,
        test_data,
        steps_per_episode=INITIAL_STEPS,
        num_rollouts=20,
        horizon=30,
        seed=123,
    )

    # --------------------------------------------------------
    # 4. Common initial states
    # --------------------------------------------------------

    rng = np.random.default_rng(
        SEED_INITIAL_STATES
    )

    initial_states = np.array(
        [
            [
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-2.0, 2.0),
            ]
            for _ in range(EVAL_EPISODES)
        ],
        dtype=np.float32,
    )

    print(
        "\nInitial states used for MPPI comparison:"
    )

    for i, x0 in enumerate(initial_states):
        print(
            f"Episode {i + 1:2d}: "
            f"theta={x0[0]: .3f}, "
            f"omega={x0[1]: .3f}"
        )

    # --------------------------------------------------------
    # 5. Single learned-model MPPI rollout
    # --------------------------------------------------------

    print("\n==============================")
    print("LEARNED-MODEL MPPI")
    print("==============================")

    x_history, u_history, model_errors = run_mppi(
        dynamics_model=model,
        x0=np.array([0.0, 0.0]),
        num_steps=EVAL_STEPS,
        seed=0,
    )

    theta_final = x_history[-1, 0]
    omega_final = x_history[-1, 1]

    print(
        f"Final theta       : {theta_final:.6f}"
    )
    print(
        f"Final omega       : {omega_final:.6f}"
    )
    print(
        f"Final theta error : "
        f"{angle_error(theta_final):.6f}"
    )
    print(
        f"Mean |u|          : "
        f"{np.mean(np.abs(u_history)):.6f}"
    )
    print(
        f"Max |u|           : "
        f"{np.max(np.abs(u_history)):.6f}"
    )

    plot_trajectory(
        x_history,
        u_history,
    )

    plot_model_error(
        model_errors,
    )

    # --------------------------------------------------------
    # 6. Perfect-model MPPI
    # --------------------------------------------------------

    print("\n==============================")
    print("PERFECT-MODEL MPPI")
    print("==============================")

    perfect_model = AnalyticalDynamicsModel()

    perfect_results = evaluate_mppi(
        dynamics_model=perfect_model,
        initial_states=initial_states,
        num_steps=EVAL_STEPS,
        seed=100,
    )

    summarize_results(
        perfect_results,
        "PERFECT-MODEL MPPI EVALUATION",
    )

    # --------------------------------------------------------
    # 7. Fixed learned-model MPPI
    # --------------------------------------------------------

    print("\n==============================")
    print("LEARNED-MODEL MPPI")
    print("==============================")

    learned_results = evaluate_mppi(
        dynamics_model=model,
        initial_states=initial_states,
        num_steps=EVAL_STEPS,
        seed=100,
    )

    summarize_results(
        learned_results,
        "LEARNED-MODEL MPPI EVALUATION",
    )

    compare_model_error(
        learned_results,
        "FIXED LEARNED MODEL: SUCCESS vs FAILURE",
    )

    inspect_failed_episodes(
        learned_results
    )

    print("\n==============================")
    print("MPPI IMAGINATION DIAGNOSTIC")
    print("==============================")

    for episode, x0 in enumerate(initial_states):

        np.random.seed(500 + episode)

        controller = MPPIController(
            dynamics_model=model
        )

        u0, imagined_errors = controller.command(
            x0,
            diagnose=True
        )

        theta_rmse = np.sqrt(
            np.mean(imagined_errors[:, 0] ** 2)
        )

        omega_rmse = np.sqrt(
            np.mean(imagined_errors[:, 1] ** 2)
        )

        total_rmse = np.sqrt(
            np.mean(imagined_errors ** 2)
        )

        print(
            f"Episode {episode + 1:2d}: "
            f"theta={theta_rmse:.4f} | "
            f"omega={omega_rmse:.4f} | "
            f"total={total_rmse:.4f}"
        )

    # --------------------------------------------------------
    # 8. Online MBRL
    # --------------------------------------------------------

    print("\n==============================")
    print("ONLINE MBRL")
    print("==============================")

    online_model, online_results = run_online_mbrl(
        initial_data=data,
        initial_states=initial_states,
        num_episodes=ONLINE_EPISODES,
        train_epochs=ONLINE_TRAIN_EPOCHS,
    )
