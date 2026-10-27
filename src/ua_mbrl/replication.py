"""Compare a suite run against the paper's published numbers.

    ua-mbrl check results/<tag>/replicate-<timestamp>

Writes replication_report.md / .json into the suite directory. Seeded
outcomes (success rates, paired records, AUCs, per-seed MuJoCo records) are
expected to match exactly on the hardware that produced them (GTX 1650);
on other GPUs floating-point differences can legitimately change individual
episodes, so a mismatch there is a finding to report, not a bug. Latency
results are hardware-dependent and reported for information only.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ua_mbrl.utils.results import PUBLISHED_DIR, read_jsonl


@dataclass
class Check:
    experiment: str
    item: str
    expected: object
    observed: object
    status: str  # "match" | "mismatch" | "info" | "missing"


def _cmp(experiment, item, expected, observed, tol=0.0):
    if observed is None:
        return Check(experiment, item, expected, None, "missing")
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        ok = abs(expected - observed) <= tol
    else:
        ok = expected == observed
    return Check(experiment, item, expected, observed, "match" if ok else "mismatch")


# --- published targets -----------------------------------------------------

CLAIM_A = {  # sec:claima: matched %, matched K, adaptive %, fixed wins, adaptive wins, ties
    (3000, "nominal"): (92.5, 299, 82.5, 7, 3, 30),
    (8000, "nominal"): (97.5, 403, 82.5, 7, 1, 32),
    (16000, "nominal"): (100.0, 682, 82.5, 7, 0, 33),
    (1000, "mild_mismatch"): (75.0, 284, 50.0, 14, 4, 22),
}
CONFOUND_FIXED = {100: (0.75, 0.75), 1000: (0.90, 0.75), 8000: (0.70, 0.70)}  # tab:confoundfixed corrected
CONFOUND_PREFIX = {100: (0.75, 0.35), 1000: (0.40, 0.35), 8000: (0.45, 0.40)}  # tab:confound (pre-fix controller)
OOD_AUC = [0.520, 0.409, 0.458, 0.551, 0.654, 0.764, 0.895, 0.979, 0.999]  # fig1
CROSSOVER_K = 15600  # sec:crossover, GTX 1650


def check_claim_a(records):
    out = []
    by_case = {(r["k_max"], r["condition"]): r for r in records}
    for case, (matched, k, adaptive, fw, aw, ties) in CLAIM_A.items():
        r = by_case.get(case)
        name = f"K_max={case[0]} {case[1]}"
        if r is None:
            out.append(Check("claim_a", name, "case", None, "missing"))
            continue
        out += [
            _cmp("claim_a", f"{name}: fixed@matched %", matched, r["matched_success"]),
            _cmp("claim_a", f"{name}: matched K", k, r["matched_k"]),
            _cmp("claim_a", f"{name}: adaptive %", adaptive, r["adaptive_success"]),
            _cmp("claim_a", f"{name}: paired (fixed/adaptive/tie)", [fw, aw, ties],
                 [r["fixed_wins"], r["adaptive_wins"], r["ties"]]),
        ]
    return out


def check_crossover_quality(records):
    out = []
    for r in records:
        expected = CONFOUND_FIXED.get(r["k_max"])
        if expected:
            out.append(_cmp("crossover_quality", f"K_max={r['k_max']} corrected fixed/adaptive", list(expected),
                            [r["retuned"]["fixed_success"], r["retuned"]["adaptive_success"]]))
        out.append(Check("crossover_quality", f"K_max={r['k_max']} sigma=1.0 fixed/adaptive (current controller; "
                         "tab:confound itself is reproduced by confound_check --pilot-rng shared)", "n/a",
                         [r["original"]["fixed_success"], r["original"]["adaptive_success"]], "info"))
    return out


def check_confound(records):
    return [_cmp("confound_check", f"K_max={r['k_max']} fixed/adaptive (tab:confound, --pilot-rng shared)",
                 list(CONFOUND_PREFIX[r["k_max"]]), [r["fixed_success"], r["adaptive_success"]])
            for r in records if r["k_max"] in CONFOUND_PREFIX]


def check_stratified(records):
    part2 = {(r["condition"], r["method"]): np.mean(r["per_seed"]) for r in records if r.get("part") == 2}
    conditions = sorted({c for c, _ in part2})
    low_beats_high = sum(part2[(c, "fixed_low")] > part2[(c, "fixed_high")] for c in conditions)
    return [_cmp("claim_a_stratified", "conditions where K=100 beats K=300 (n=15)", 2, int(low_beats_high))]


def check_fallback(records):
    out = []
    mismatch = [r for r in records if r.get("variant") == "mismatch" and r.get("method") == "fallback"]
    if mismatch:
        out.append(_cmp("fallback", "mismatch: fallback never triggers (max trigger %)", 0.0,
                        max(100 * r["fallback_fraction"] for r in mismatch)))
    for variant, expected in (("disturbance", 0.0), ("narrow", 8.7)):
        rows = [r for r in records if r.get("variant") == variant and r.get("method") == "fallback"]
        if rows:
            out.append(_cmp("fallback", f"{variant}: fallback trigger % of steps", expected,
                            round(100 * rows[0]["fallback_fraction"], 1)))
    return out


def check_ood(records):
    observed = [round(r["auc_mean"], 3) for r in sorted(records, key=lambda r: r["shift"])]
    return [_cmp("ood_severity", "AUC per shift level (fig1)", OOD_AUC, observed)]


def _per_seed(experiment, records, published, fields):
    out = []
    published = {r["seed"]: r for r in published}
    for r in records:
        p = published.get(r["seed"])
        if p is None:
            continue
        mismatched = [f for f in fields if _get(r, f) != _get(p, f)]
        out.append(Check(experiment, f"seed {r['seed']}", "identical to published record",
                         "identical" if not mismatched else f"differs: {', '.join(mismatched)}",
                         "match" if not mismatched else "mismatch"))
    return out


def _get(record, dotted):
    value = record
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {str(k): round(v, 9) if isinstance(v, float) else v for k, v in value.items()}
    return value


METHOD_FIELDS = [f"{m}.{k}" for m in ("single", "ensemble", "adaptive") for k in ("success_rate", "effective_rollouts")]


def check_mujoco_sigma1(records):
    return _per_seed("mujoco_regulation sigma=1.0", records, read_jsonl(PUBLISHED_DIR / "mujoco_multiseed_results.jsonl"),
                     ["pd_success", "threshold", *METHOD_FIELDS])


def check_mujoco_sigma02(records):
    return _per_seed("mujoco_regulation sigma=0.2", records,
                     read_jsonl(PUBLISHED_DIR / "mujoco_quality_retuned_results.jsonl"), ["threshold", *METHOD_FIELDS])


def check_mujoco_claim_a(records):
    return _per_seed("mujoco_claim_a", records, read_jsonl(PUBLISHED_DIR / "mujoco_claim_a_results.jsonl"),
                     ["k_matched", "fixed_matched_sigma_sweep", "fixed_matched_best_sigma", "fixed_matched_best_success"])


def check_reacher(records):
    return _per_seed("reacher", records, read_jsonl(PUBLISHED_DIR / "mujoco_reacher_results.jsonl"),
                     [f"{m}.{k}" for m in ("single", "ensemble", "adaptive") for k in ("success_rate", "final_distance")])


def check_crossover_sweep(suite_label_dir):
    analysis_path = suite_label_dir / "analysis.json"
    if not analysis_path.exists():
        return [Check("crossover_sweep", "analysis.json", "present", None, "missing")]
    analysis = json.loads(analysis_path.read_text())
    published = {r["k_max"]: r["ratio"] for r in read_jsonl(PUBLISHED_DIR / "adaptive_k_fine_results.jsonl")}
    out = [Check("crossover_sweep", "interpolated crossover K_max (hardware-dependent; GTX 1650: ~15,600)",
                 CROSSOVER_K, analysis["interpolated_crossover"], "info")]
    if analysis.get("regression"):
        out.append(Check("crossover_sweep", "regression K* (GTX 1650: 14,773)", 14773,
                         round(analysis["regression"]["k_star"]), "info"))
    for k, ratio in analysis["ratio"].items():
        if int(k) in published:
            out.append(Check("crossover_sweep", f"ratio at K_max={k}", round(published[int(k)], 3), round(ratio, 3), "info"))
    return out


CHECKS = {
    "claim_a": check_claim_a,
    "crossover_quality": check_crossover_quality,
    "confound_check": check_confound,
    "claim_a_stratified": check_stratified,
    "fallback": check_fallback,
    "ood_severity": check_ood,
    "mujoco_regulation_sigma1": check_mujoco_sigma1,
    "mujoco_regulation_sigma0.2": check_mujoco_sigma02,
    "mujoco_claim_a": check_mujoco_claim_a,
    "reacher": check_reacher,
}


def check_suite(suite_dir):
    suite_dir = Path(suite_dir)
    checks = []
    for label_dir in sorted(p for p in suite_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        if label == "crossover_sweep":
            checks += check_crossover_sweep(label_dir)
        elif label in CHECKS:
            records = read_jsonl(label_dir / "records.jsonl")
            if not (label_dir / "DONE").exists():
                checks.append(Check(label, "run", "completed", "incomplete", "missing"))
            if records:
                checks += CHECKS[label](records)
    meta_paths = sorted(suite_dir.glob("*/meta.json"))
    environment = json.loads(meta_paths[0].read_text())["environment"] if meta_paths else {}
    write_report(suite_dir, checks, environment)
    return checks


def write_report(suite_dir, checks, environment):
    counts = {s: sum(c.status == s for c in checks) for s in ("match", "mismatch", "missing", "info")}
    lines = [
        f"# Replication report: {suite_dir.name}",
        "",
        f"GPU: {environment.get('gpu_name', 'cpu')} | driver {environment.get('nvidia_driver', '?')} | "
        f"torch {environment.get('torch', '?')} (CUDA {environment.get('torch_cuda', '?')}) | "
        f"commit {environment.get('git_commit', '?')[:10]}",
        "",
        f"**{counts['match']} match, {counts['mismatch']} mismatch, {counts['missing']} missing, {counts['info']} info**",
        "",
        "| experiment | item | expected | observed | status |",
        "|---|---|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| {c.experiment} | {c.item} | {c.expected} | {c.observed} | {c.status} |")
    (suite_dir / "replication_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (suite_dir / "replication_report.json").write_text(
        json.dumps({"environment": environment, "counts": counts, "checks": [asdict(c) for c in checks]},
                   indent=2, default=str), encoding="utf-8")
    print("\n".join(lines))
