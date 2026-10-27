"""Refactor equivalence: the ua_mbrl package must reproduce the frozen legacy
scripts (legacy/) exactly, at reduced scale.

Each test re-enacts the legacy script's ``__main__`` sequence with the legacy
functions (module constants shrunk with monkeypatch) and runs the package's
experiment at the same scale, then requires *exact* equality of every
seeded outcome (success per seed, thresholds, effective K, AUCs, ...).
Wall-clock latencies are not compared.

Both sides train their own models, so each starts from the same RNG state;
any reordering of RNG consumption in the refactor shows up as a mismatch.
"""

import numpy as np
import pytest
import torch

from ua_mbrl.envs import pendulum as pend
from ua_mbrl.experiments import build_args
from ua_mbrl.utils.device import DEVICE
from ua_mbrl.utils.results import PUBLISHED_DIR, read_jsonl

from conftest import assert_same

pytestmark = pytest.mark.slow


def _params(model_list):
    return [p.detach().cpu().numpy() for m in model_list for p in m.parameters()]


def test_pendulum_training_identical(legacy):
    rme = legacy("robustness_mismatch_experiment")
    single_old, ensemble_old = rme.train_models(rme.collect_data())
    single_new, ensemble_new = pend.train_models(pend.collect_data())
    assert_same(_params([single_old]), _params([single_new]))
    assert_same(_params(ensemble_old.models), _params(ensemble_new.models))


def test_narrow_training_identical(legacy):
    ood = legacy("ood_uncertainty_experiment")
    old = ood.train_models(ood.collect_narrow_data())
    new = pend.train_models(pend.collect_narrow_data(), train_fraction=None)
    assert_same(_params([old[0]] + old[1].models), _params([new[0]] + new[1].models))


def test_claim_a_case(legacy, monkeypatch, run_dir):
    from ua_mbrl.experiments import claim_a

    rme = legacy("robustness_mismatch_experiment")
    ext = legacy("claim_a_extended")
    paired = legacy("claim_a_paired_test")
    monkeypatch.setattr(ext, "EVAL_STEPS", 6)
    monkeypatch.setattr(paired, "EVAL_STEPS", 6)
    monkeypatch.setattr(paired, "CEILING_SIGMAS", (1.0, 3.0))
    monkeypatch.setattr(ext, "CEILING_SIGMAS", (1.0, 3.0))
    calibration_states = ext.make_states(900, 40)
    states = ext.make_states(800, 8)

    # Aggregates (claim_a_extended) and paired record (claim_a_paired_test).
    _, ensemble = rme.train_models(rme.collect_data())
    old_ext = ext.run_test_case(ensemble, calibration_states, states, 600, {"mass": 1.2}, "mild_mismatch")
    _, ensemble = rme.train_models(rme.collect_data())
    old_paired = paired.run_test_case(ensemble, calibration_states, states, 600, {"mass": 1.2}, "mild_mismatch")

    args = build_args("claim_a", eval_steps=6, sigmas=[1.0, 3.0])
    new = claim_a.run_case(pend.train_default_ensemble(), calibration_states, states, 600, "mild_mismatch", args)

    for key in ("ceiling_sigma", "ceiling_success", "adaptive_success", "eff_k", "matched_k", "matched_sigma", "matched_success"):
        assert new[key] == old_ext[key], key
    for key in ("fixed_wins", "adaptive_wins", "ties", "sign_test_p", "matched_success", "adaptive_success"):
        assert new[key] == old_paired[key] or (np.isnan(new[key]) and np.isnan(old_paired[key])), key


def test_crossover_point(legacy):
    from ua_mbrl.experiments import crossover

    rme = legacy("robustness_mismatch_experiment")
    cs = legacy("adaptive_k_crossover_study")
    calibration_states = cs.make_initial_states(900, 40)
    states = cs.make_initial_states(800, 6)

    _, ensemble = rme.train_models(rme.collect_data())
    threshold = cs.calibrate_threshold(ensemble, 2000, calibration_states)
    fixed = cs.run_fixed_high_batched(ensemble, states, 2000, 5, seed=1000)
    adaptive = cs.run_adaptive_sequential(ensemble, states, 2000, threshold, 5, seed=2000)

    ensemble = pend.train_default_ensemble()
    new_threshold = pend.calibrate_threshold(ensemble, 2000, calibration_states)
    record = crossover.measure(ensemble, calibration_states, states, 2000, new_threshold, 5)

    assert new_threshold == threshold
    assert record["fixed_success"] == fixed["success"]
    assert record["adaptive_success"] == adaptive["success"]
    assert record["adaptive_eff_k"] == adaptive["effective_rollouts"]


def test_confound_check(legacy, run_dir):
    from ua_mbrl.experiments import crossover_quality

    rme = legacy("robustness_mismatch_experiment")
    cs = legacy("adaptive_k_crossover_study")
    acc = legacy("adaptive_k_confound_check")
    _, ensemble = rme.train_models(rme.collect_data())
    calibration_states = cs.make_initial_states(900, 40)
    states = cs.make_initial_states(800, 6)
    old = []
    for k_max in (100, 1000):
        threshold = cs.calibrate_threshold(ensemble, k_max, calibration_states)
        old.append((
            acc.run_fixed_high_sequential(ensemble, states, k_max, 5, seed=1000),
            acc.run_adaptive_sequential_success_only(ensemble, states, k_max, threshold, 5, seed=1000),
        ))

    crossover_quality.run_confound(build_args("confound_check", k_values=[100, 1000], num_seeds=6, eval_steps=5), run_dir)
    new = [(r["fixed_success"], r["adaptive_success"]) for r in run_dir.records()]
    assert new == old


def test_crossover_quality_retuned(legacy, monkeypatch, run_dir):
    from ua_mbrl.experiments import crossover_quality

    rme = legacy("robustness_mismatch_experiment")
    cqr = legacy("crossover_quality_retuned")
    monkeypatch.setattr(cqr, "EVAL_STEPS", 5)
    _, ensemble = rme.train_models(rme.collect_data())
    calibration_states = cqr.make_initial_states(900, 40)
    states = cqr.make_initial_states(800, 6)
    old = []
    for k_max in (100, 1000):
        best = cqr.BEST_SIGMA[k_max]
        th_o = cqr.make_adaptive(ensemble, k_max, 0.0, 1.0, 1.0).calibrate_threshold(calibration_states)
        fo = cqr.run_success(lambda: cqr.make_fixed(ensemble, k_max, 1.0), states, seed=1000)
        ao = cqr.run_success(lambda: cqr.make_adaptive(ensemble, k_max, th_o, 1.0, 1.0), states, seed=1000)
        th_r = cqr.make_adaptive(ensemble, k_max, 0.0, 1.0, best).calibrate_threshold(calibration_states)
        fr = cqr.run_success(lambda: cqr.make_fixed(ensemble, k_max, best), states, seed=1000)
        ar = cqr.run_success(lambda: cqr.make_adaptive(ensemble, k_max, th_r, 1.0, best), states, seed=1000)
        old.append((th_o, fo, ao, th_r, fr, ar))

    crossover_quality.run(build_args("crossover_quality", k_values=[100, 1000], num_seeds=6, eval_steps=5), run_dir)
    new = [
        (r["original"]["threshold"], r["original"]["fixed_success"], r["original"]["adaptive_success"],
         r["retuned"]["threshold"], r["retuned"]["fixed_success"], r["retuned"]["adaptive_success"])
        for r in run_dir.records()
    ]
    assert new == old


def test_claim_a_stratified(legacy, monkeypatch, run_dir):
    from ua_mbrl.experiments import claim_a_stratified

    rme = legacy("robustness_mismatch_experiment")
    cse = legacy("claim_a_stratified_experiment")
    steps, num_seeds = 12, 3
    monkeypatch.setattr(cse, "EVAL_STEPS", steps)

    # --- legacy __main__, re-enacted ---
    _, ensemble = rme.train_models(rme.collect_data())
    threshold = cse.make_adaptive(ensemble, threshold=0.0).calibrate_threshold(cse.make_states(900, 40))
    ref = cse.make_states(700, 1)
    old_rows = []
    for name, condition in cse.CONDITIONS.items():
        old_rows += cse.run_reference_trajectory(ensemble, threshold, name, condition, ref[0], seed=500, disturb=False)
    old_rows += cse.run_reference_trajectory(ensemble, threshold, "state_disturbance", {}, ref[0], seed=500, disturb=True)

    states = cse.make_states(800, num_seeds)
    conditions = dict(cse.CONDITIONS)
    conditions["state_disturbance"] = {}
    old_part2 = {}
    for condition_name, condition in conditions.items():
        for i in range(num_seeds):
            seed, x0 = 1000 + i, states[i]

            def run_with(make_ctrl, disturb=(condition_name == "state_disturbance")):
                np.random.seed(seed)
                torch.manual_seed(seed)
                controller = make_ctrl()
                controller.reset()
                x = x0.copy()
                for step in range(steps):
                    if disturb and step == cse.DISTURBANCE_STEP:
                        x[1] = cse.DISTURBANCE_OMEGA
                    action = controller.command(x)
                    x = cse.pendulum_step(x, float(np.asarray(action).reshape(-1)[0]), **condition)
                return abs(cse.angle_error(x[0])) < 0.1 and abs(x[1]) < 0.5

            for name, factory in (
                ("fixed_low", lambda: cse.make_fixed(ensemble, cse.K_MIN)),
                ("fixed_high", lambda: cse.make_fixed(ensemble, cse.K_MAX)),
                ("adaptive", lambda: cse.make_adaptive(ensemble, threshold)),
            ):
                old_part2.setdefault((condition_name, name), []).append(bool(run_with(factory)))

    # --- package ---
    claim_a_stratified.run(build_args("claim_a_stratified", num_seeds=num_seeds, eval_steps=steps), run_dir)
    records = run_dir.records()
    new_rows = [{k: r[k] for k in ("condition", "step", "eta", "k_used", "ess_min", "ess_max")}
                for r in records if r["part"] == 1]
    assert_same(old_rows, new_rows)
    new_part2 = {(r["condition"], r["method"]): r["per_seed"] for r in records if r["part"] == 2}
    assert new_part2 == old_part2


def _fallback_old_controllers(cfe, ensemble, mild, severe):
    common = dict(
        dynamics=ensemble, action_dim=1, horizon=cfe.HORIZON, action_low=[cfe.U_MIN], action_high=[cfe.U_MAX],
        target=[np.pi, 0.0], cost_fn=cfe.pendulum_cost, dt=cfe.DT, temperature=1.0, noise_sigma=1.0,
        min_rollouts=cfe.ROLLOUTS, max_rollouts=3 * cfe.ROLLOUTS, uncertainty_threshold=mild,
        pilot_rollouts=25, pilot_horizon=5, device=cfe.DEVICE,
    )
    return {
        "adaptive": lambda: cfe.AdaptiveGenericMPPI(**common),
        "fallback": lambda: cfe.FallbackAdaptiveMPPI(**common, severe_threshold=severe),
    }


def test_fallback_mismatch(legacy, monkeypatch, run_dir):
    from ua_mbrl.experiments import fallback

    rme = legacy("robustness_mismatch_experiment")
    cfe = legacy("conservative_fallback_experiment")
    monkeypatch.setattr(cfe, "EVAL_STEPS", 8)
    _, ensemble = rme.train_models(rme.collect_data())
    mild = cfe.make_calibrator(ensemble).calibrate_threshold(cfe.make_states(900, 40))
    severe = mild * cfe.SEVERE_MULTIPLIER
    states = cfe.make_states(800, 4)
    conditions = [{}, {"mass": 1.2}, {"mass": 3.0, "gravity": cfe.NOMINAL_G * 0.4}]
    old = []
    for condition in conditions:
        for factory in _fallback_old_controllers(cfe, ensemble, mild, severe).values():
            old.append(cfe.evaluate(factory, states, condition, seed_base=1000))

    fallback.run(build_args("fallback", variants=["mismatch"], num_seeds=4, eval_steps=8), run_dir)
    new = [{k: r[k] for k in ("success", "max_abs_omega_mean", "max_abs_omega_worst", "fallback_fraction")}
           for r in run_dir.records()]
    assert_same(old, new)


@pytest.mark.parametrize("variant", ["disturbance", "narrow"])
def test_fallback_disturbance(legacy, monkeypatch, run_dir, variant):
    from ua_mbrl.experiments import fallback

    rme = legacy("robustness_mismatch_experiment")
    cfe = legacy("conservative_fallback_experiment")
    module = legacy("conservative_fallback_disturbance_test" if variant == "disturbance"
                    else "conservative_fallback_narrow_ensemble")
    monkeypatch.setattr(module, "EVAL_STEPS", 14)
    if variant == "disturbance":
        _, ensemble = rme.train_models(rme.collect_data())
        mild = cfe.make_calibrator(ensemble).calibrate_threshold(cfe.make_states(900, 40))
        states = cfe.make_states(800, 4)
    else:
        _, ensemble = module.train_narrow_models(module.collect_narrow_data(seed=0))
        calibrator = cfe.make_calibrator(ensemble)
        clip = ([-np.pi, -1.0], [np.pi, 1.0])
        mild = calibrator.calibrate_threshold(np.clip(cfe.make_states(900, 40), *clip).astype(np.float32))
        states = np.clip(cfe.make_states(800, 4), *clip).astype(np.float32)
    severe = mild * 3.5
    old = []
    for factory in _fallback_old_controllers(cfe, ensemble, mild, severe).values():
        results = [module.run_episode_with_disturbance(factory(), x0, 1000 + i, track_fallback=True)
                   for i, x0 in enumerate(states)]
        old.append({
            "success": float(np.mean([r["success"] for r in results])),
            "max_abs_omega_mean": float(np.mean([r["max_abs_omega_after"] for r in results])),
            "max_abs_omega_worst": float(np.max([r["max_abs_omega_after"] for r in results])),
            "fallback_fraction": float(np.mean([r["fallback_fraction"] for r in results])),
        })

    fallback.run(build_args("fallback", variants=[variant], num_seeds=4, eval_steps=14), run_dir)
    new = [{k: r[k] for k in old[0]} for r in run_dir.records() if "method" in r]
    assert_same(old, new)


def test_ood_severity(legacy, monkeypatch, run_dir):
    from ua_mbrl.experiments import ood

    oss = legacy("ood_severity_sweep")
    monkeypatch.setattr(oss, "SHIFT_LEVELS", (0.0, 2.0, 3.5))
    monkeypatch.setattr(oss, "REPEATS_PER_LEVEL", 2)
    single, ensemble = oss.train_models(oss.collect_narrow_data(seed=0))
    old = oss.run_sweep(single, ensemble)

    ood.run_severity(build_args("ood_severity", shift_levels=[0.0, 2.0, 3.5], repeats=2), run_dir)
    new = [{k: r[k] for k in old[0]} for r in run_dir.records()]
    assert_same(old, new)


def test_robustness(legacy, monkeypatch, run_dir):
    from ua_mbrl.experiments import robustness

    rme = legacy("robustness_mismatch_experiment")
    monkeypatch.setattr(rme, "EVAL_STEPS", 6)
    monkeypatch.setattr(rme, "EVAL_EPISODES", 2)
    single, ensemble = rme.train_models(rme.collect_data())
    calibration_states, states = rme.initial_states(seed=101), rme.initial_states(seed=100)
    calibrator = rme.adaptive_controller_factory(ensemble)()
    torch.manual_seed(101)
    threshold = calibrator.calibrate_threshold(calibration_states)
    methods = [
        rme.controller_factory(rme.AnalyticalNominalModel()), rme.controller_factory(single),
        rme.controller_factory(ensemble), rme.controller_factory(ensemble, rollouts=300),
        rme.controller_factory(ensemble, beta=0.1), rme.adaptive_controller_factory(ensemble, threshold=threshold),
    ]
    old = [rme.evaluate(factory, states, params, seed_offset=1000)
           for params in ({}, {"mass": 1.2}, {"gravity": 9.81 * 0.85}) for factory in methods]

    robustness.run(build_args("robustness", eval_steps=6, episodes=2), run_dir)
    new = [{k: r[k] for k in old[0]} for r in run_dir.records()]
    assert_same(old, new)


MUJOCO_KEYS = ("success_rate", "effective_rollouts")


@pytest.mark.parametrize("sigma", [1.0, 0.2])
def test_mujoco_regulation(legacy, monkeypatch, sigma):
    from ua_mbrl.experiments import mujoco_regulation

    mb = legacy("mujoco_benchmark")
    monkeypatch.setattr(mb, "STEPS", 10)
    monkeypatch.setattr(mb, "EPISODES", 2)
    if sigma == 1.0:
        old = legacy("mujoco_multiseed_study").run_one_seed(7, mb.DEVICE)
    else:
        old = legacy("mujoco_quality_retuned").run_one_seed(7, mb.DEVICE)

    args = build_args("mujoco_regulation", sigma=sigma, steps=10, episodes=2)
    new = mujoco_regulation.run_one_seed(7, DEVICE, args)
    assert new["threshold"] == old["threshold"]
    if "pd_success" in old:
        assert new["pd_success"] == old["pd_success"]
    for method in ("single", "ensemble", "adaptive"):
        for key in MUJOCO_KEYS:
            assert new[method][key] == old[method][key], (method, key)


def test_mujoco_claim_a(legacy, monkeypatch):
    from ua_mbrl.experiments import mujoco_claim_a

    mb = legacy("mujoco_benchmark")
    mca = legacy("mujoco_claim_a")
    monkeypatch.setattr(mb, "STEPS", 10)
    monkeypatch.setattr(mb, "EPISODES", 2)
    monkeypatch.setattr(mca, "SIGMAS", (0.1, 0.25))
    original = {r["seed"]: r for r in read_jsonl(PUBLISHED_DIR / "mujoco_quality_retuned_results.jsonl")}
    old = mca.run_one_seed(17, mb.DEVICE, original[17])

    args = build_args("mujoco_claim_a", sigmas=[0.1, 0.25], steps=10, episodes=2)
    new = mujoco_claim_a.run_one_seed(17, DEVICE, original[17], args)
    assert_same({str(k): v for k, v in old.items()}, {str(k): v for k, v in new.items()})


def test_reacher(legacy, monkeypatch):
    from ua_mbrl.experiments import reacher

    mre = legacy("mujoco_reacher_experiment")
    monkeypatch.setattr(mre, "STEPS", 12)
    monkeypatch.setattr(mre, "EPISODES", 2)

    # legacy __main__ body for one seed
    seed, device = 7, mre.DEVICE
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_data = mre.collect_task_directed_data(mre.ReacherMuJoCo(), transitions=2000, seed=seed)
    episode_seeds = mre.make_episode_seeds(seed + 1000)
    normalizer = mre.make_normalizer(train_data, device)
    torch.manual_seed(seed)
    single = mre.train_single(train_data, device, normalizer)
    torch.manual_seed(seed)
    ensemble = mre.train_ensemble(train_data, device, normalizer)
    calibrator = mre.make_controller(ensemble, device, np.zeros(2, dtype=np.float32), adaptive=True)
    torch.manual_seed(seed + 5000)
    threshold = calibrator.calibrate_threshold([mre.ReacherMuJoCo().reset(seed=s) for s in mre.make_episode_seeds(seed + 2000)])
    old = {"threshold": threshold}
    for name, model, is_adaptive in [("single", single, False), ("ensemble", ensemble, False), ("adaptive", ensemble, True)]:
        old[name] = mre.evaluate(model, device, episode_seeds, is_adaptive, threshold)

    new = reacher.run_one_seed(seed, DEVICE, build_args("reacher", steps=12, episodes=2))
    assert new["threshold"] == old["threshold"]
    for method in ("single", "ensemble", "adaptive"):
        for key in ("success_rate", "final_distance"):
            assert new[method][key] == old[method][key], (method, key)


def test_shared_pilot_rng_reproduces_published_mujoco_record():
    """tab:mujoco predates the pilot-RNG isolation fix; --pilot-rng shared
    (the pre-fix controller) must reproduce its stored per-seed record."""
    from ua_mbrl.core.mppi import set_default_pilot_rng
    from ua_mbrl.experiments import mujoco_regulation

    published = {r["seed"]: r for r in read_jsonl(PUBLISHED_DIR / "mujoco_multiseed_results.jsonl")}[7]
    set_default_pilot_rng("shared")
    try:
        record = mujoco_regulation.run_one_seed(7, DEVICE, build_args("mujoco_regulation"))
    finally:
        set_default_pilot_rng("isolated")
    assert record["threshold"] == pytest.approx(published["threshold"], rel=1e-9)
    assert record["pd_success"] == published["pd_success"]
    for method in ("single", "ensemble", "adaptive"):
        for key in MUJOCO_KEYS:
            assert record[method][key] == published[method][key], (method, key)
