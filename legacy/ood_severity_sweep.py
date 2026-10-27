"""Continuous OOD-severity sweep.

ood_uncertainty_experiment.py showed a binary asymmetry: severe OOD is
detected almost perfectly (AUC ~1.000) while mild OOD is detected only
moderately well (AUC ~0.630) -- but that was tested at exactly two discrete
points with a single test set each. This script sweeps distribution-shift
magnitude continuously and reports AUC / TPR / FPR / uncertainty-ratio as a
function of shift magnitude, with confidence intervals from repeated
independent (in-distribution, OOD) test-set pairs at each shift level,
against one fixed trained ensemble (training happens once, up front).
"""

import numpy as np
import torch

from ood_uncertainty_experiment import (
    DEVICE,
    collect_narrow_data,
    evaluate,
    evaluate_ood_detection,
    make_test_data,
    train_models,
)

SHIFT_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
REPEATS_PER_LEVEL = 15
IN_DIST_OMEGA_RANGE = (-1.0, 1.0)
IN_DIST_ACTION_RANGE = (-2.0, 2.0)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean()
    ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, ci


def run_sweep(single, ensemble):
    rows = []
    for shift in SHIFT_LEVELS:
        aucs, tprs, fprs, unc_ratios, rmse_ratios = [], [], [], [], []
        for repeat in range(REPEATS_PER_LEVEL):
            in_seed = 10_000 + repeat
            ood_seed = 20_000 + int(round(shift * 10)) * 1000 + repeat

            in_data = make_test_data(in_seed, IN_DIST_OMEGA_RANGE, IN_DIST_ACTION_RANGE)
            ood_data = make_test_data(ood_seed, (shift, shift + 1.0), (shift, shift + 1.0))

            in_result = evaluate(single, ensemble, in_data)
            ood_result = evaluate(single, ensemble, ood_data)
            detection = evaluate_ood_detection(in_result, ood_result)

            aucs.append(detection["auc"])
            tprs.append(detection["tpr"])
            fprs.append(detection["fpr"])
            unc_ratios.append(ood_result["mean_uncertainty"] / in_result["mean_uncertainty"])
            rmse_ratios.append(ood_result["ensemble_rmse"] / in_result["ensemble_rmse"])

        auc_mean, auc_ci = summarize(aucs)
        tpr_mean, tpr_ci = summarize(tprs)
        fpr_mean, fpr_ci = summarize(fprs)
        unc_mean, unc_ci = summarize(unc_ratios)
        rmse_mean, rmse_ci = summarize(rmse_ratios)

        rows.append(
            {
                "shift": shift,
                "auc_mean": auc_mean, "auc_ci": auc_ci,
                "tpr_mean": tpr_mean, "tpr_ci": tpr_ci,
                "fpr_mean": fpr_mean, "fpr_ci": fpr_ci,
                "unc_ratio_mean": unc_mean, "unc_ratio_ci": unc_ci,
                "rmse_ratio_mean": rmse_mean, "rmse_ratio_ci": rmse_ci,
            }
        )
    return rows


if __name__ == "__main__":
    print(f"Compute device: {DEVICE}")
    print(f"Shift levels: {SHIFT_LEVELS}, repeats per level: {REPEATS_PER_LEVEL}")

    train_data = collect_narrow_data(seed=0)
    single, ensemble = train_models(train_data)

    rows = run_sweep(single, ensemble)

    print("\nOOD severity sweep (95% CI over {} repeats per shift level)".format(REPEATS_PER_LEVEL))
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

    # Find where AUC crosses into "strong" (>0.9) and "near-chance" (<0.6) territory.
    strong = [r["shift"] for r in rows if r["auc_mean"] >= 0.9]
    weak = [r["shift"] for r in rows if r["auc_mean"] < 0.6]
    print("\nShift levels with AUC >= 0.9 (strong detection):", strong if strong else "none")
    print("Shift levels with AUC <  0.6 (near chance)      :", weak if weak else "none")
