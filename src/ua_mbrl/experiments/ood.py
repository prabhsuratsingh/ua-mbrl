"""Ensemble disagreement as an OOD detector (narrow-data pendulum ensemble).

``ood_severity``: AUC / TPR / FPR / uncertainty ratio vs shift magnitude,
REPEATS independent (in-distribution, OOD) test-set pairs per level, one
fixed ensemble (ood_severity_sweep.py).

``ood_two_point``: the original mild-vs-severe comparison
(ood_uncertainty_experiment.py).
"""

from ua_mbrl.envs import pendulum as pend
from ua_mbrl.eval.ood import evaluate, evaluate_ood_detection
from ua_mbrl.utils.stats import summarize_z

SHIFT_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
IN_DIST_OMEGA_RANGE = (-1.0, 1.0)
IN_DIST_ACTION_RANGE = (-2.0, 2.0)


def add_severity_arguments(parser):
    parser.add_argument("--shift-levels", type=float, nargs="+", default=list(SHIFT_LEVELS))
    parser.add_argument("--repeats", type=int, default=15)


def add_two_point_arguments(parser):
    pass


def _train():
    return pend.train_models(pend.collect_narrow_data(seed=0), train_fraction=None)


def run_severity(args, run_dir):
    single, ensemble = _train()
    rows = []
    for shift in args.shift_levels:
        metrics = {"auc": [], "tpr": [], "fpr": [], "unc_ratio": [], "rmse_ratio": []}
        for repeat in range(args.repeats):
            in_seed = 10_000 + repeat
            ood_seed = 20_000 + int(round(shift * 10)) * 1000 + repeat
            in_data = pend.make_test_data(in_seed, IN_DIST_OMEGA_RANGE, IN_DIST_ACTION_RANGE)
            ood_data = pend.make_test_data(ood_seed, (shift, shift + 1.0), (shift, shift + 1.0))
            in_result = evaluate(single, ensemble, in_data)
            ood_result = evaluate(single, ensemble, ood_data)
            detection = evaluate_ood_detection(in_result, ood_result)
            metrics["auc"].append(detection["auc"])
            metrics["tpr"].append(detection["tpr"])
            metrics["fpr"].append(detection["fpr"])
            metrics["unc_ratio"].append(ood_result["mean_uncertainty"] / in_result["mean_uncertainty"])
            metrics["rmse_ratio"].append(ood_result["ensemble_rmse"] / in_result["ensemble_rmse"])

        row = {"shift": shift, "repeats": metrics}
        for key, values in metrics.items():
            row[f"{key}_mean"], row[f"{key}_ci"] = summarize_z(values)
        run_dir.append(row)
        rows.append(row)

    print(f"\nOOD severity sweep (95% CI over {args.repeats} repeats per shift level)")
    print("=" * 100)
    print(f"{'shift':>6} {'AUC':>16} {'TPR':>16} {'FPR':>14} {'unc_ratio':>16} {'rmse_ratio':>14}")
    print("-" * 100)
    for row in rows:
        print(
            f"{row['shift']:6.1f} "
            f"{row['auc_mean']:.3f}+/-{row['auc_ci']:.3f} "
            f"{row['tpr_mean']:.3f}+/-{row['tpr_ci']:.3f} "
            f"{row['fpr_mean']:.3f}+/-{row['fpr_ci']:.3f} "
            f"{row['unc_ratio_mean']:8.2f}+/-{row['unc_ratio_ci']:.2f} "
            f"{row['rmse_ratio_mean']:6.2f}+/-{row['rmse_ratio_ci']:.2f}"
        )
    strong = [r["shift"] for r in rows if r["auc_mean"] >= 0.9]
    weak = [r["shift"] for r in rows if r["auc_mean"] < 0.6]
    print("\nShift levels with AUC >= 0.9 (strong detection):", strong or "none")
    print("Shift levels with AUC <  0.6 (near chance)      :", weak or "none")
    return rows


def run_two_point(args, run_dir):
    single, ensemble = _train()
    in_result = evaluate(single, ensemble, pend.make_test_data(10, (-1.0, 1.0), (-2.0, 2.0)))
    tests = {
        "mild OOD": pend.make_test_data(20, (2.0, 3.0), (2.0, 3.0)),
        "severe OOD": pend.make_test_data(20, (4.0, 6.0), (4.0, 5.0)),
    }
    scalar_keys = ("ensemble_rmse", "single_rmse", "mean_uncertainty", "median_uncertainty", "spearman", "top_error_capture")
    print(f"{'in-distribution':<16} " + " ".join(f"{k}={in_result[k]:.5f}" for k in scalar_keys))
    for name, data in tests.items():
        result = evaluate(single, ensemble, data)
        detection = evaluate_ood_detection(in_result, result)
        record = {
            "test": name,
            **{k: result[k] for k in scalar_keys},
            "uncertainty_ratio": result["mean_uncertainty"] / in_result["mean_uncertainty"],
            "rmse_ratio": result["ensemble_rmse"] / in_result["ensemble_rmse"],
            **detection,
        }
        run_dir.append(record)
        print(f"{name:<16} " + " ".join(f"{k}={result[k]:.5f}" for k in scalar_keys))
        print(f"  uncertainty ratio {record['uncertainty_ratio']:.2f}x | ROC AUC {detection['auc']:.3f} | "
              f"TPR {100 * detection['tpr']:.1f}% | FPR {100 * detection['fpr']:.1f}%")
