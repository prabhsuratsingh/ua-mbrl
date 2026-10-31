"""Named lists of experiment runs, executed into one directory:

    results/<tag>/<suite>-<timestamp>/<experiment-label>/...

``replicate`` reruns every paper result that has code, at published scale;
``ua-mbrl check <suite-dir>`` then compares it with the published numbers.
"""

import datetime as _dt
import subprocess
import sys
import time
from pathlib import Path

from ua_mbrl.utils.results import PUBLISHED_DIR, RESULTS_ROOT

# (label, experiment, extra argv). The label names the output subdirectory and
# is what ua_mbrl.replication looks up. Results published before the pilot-RNG
# isolation fix (tab:confound, tab:mujoco, the reacher table, the narrow-ensemble
# fallback rate) are rerun with --pilot-rng shared, i.e. the controller they
# were actually produced with; everything later uses the current controller.
SHARED = ["--pilot-rng", "shared"]
SUITES = {
    "replicate": [
        ("crossover_sweep", "crossover_sweep", []),
        ("claim_a", "claim_a", []),
        ("crossover_quality", "crossover_quality", []),
        ("confound_check", "confound_check", SHARED),
        ("claim_a_stratified", "claim_a_stratified", []),
        ("fallback", "fallback", SHARED),
        ("ood_severity", "ood_severity", []),
        ("mujoco_regulation_sigma1", "mujoco_regulation", ["--sigma", "1.0", *SHARED]),
        ("mujoco_regulation_sigma0.2", "mujoco_regulation", ["--sigma", "0.2", "--seeds", *map(str, range(7, 78, 10))]),
        ("mujoco_claim_a", "mujoco_claim_a", ["--adaptive-results", "{suite_dir}/mujoco_regulation_sigma0.2/records.jsonl"]),
        ("reacher", "reacher", SHARED),
        ("gpu_benchmark", "gpu_benchmark", ["--horizon", "10"]),
    ],
    # Fast subset: everything that has an exact published target except the
    # long MuJoCo / crossover runs.
    "replicate_quick": [
        ("claim_a", "claim_a", []),
        ("crossover_quality", "crossover_quality", []),
        ("ood_severity", "ood_severity", []),
        ("fallback", "fallback", SHARED),
        ("mujoco_claim_a", "mujoco_claim_a", ["--adaptive-results", str(PUBLISHED_DIR / "mujoco_quality_retuned_results.jsonl"),
                                               "--seeds", "7", "17"]),
    ],
    "latency": [
        ("crossover_sweep", "crossover_sweep", []),
        ("gpu_benchmark", "gpu_benchmark", ["--horizon", "10"]),
    ],
    # The four pilot-RNG-dependent experiments under the current (isolated)
    # controller: the values paper_arxiv/main.tex reports for them. Their
    # replicate-suite counterparts use the pre-fix controller instead.
    "current_controller": [
        ("confound_check", "confound_check", []),
        ("fallback", "fallback", []),
        ("mujoco_regulation_sigma1", "mujoco_regulation", ["--sigma", "1.0"]),
        ("reacher", "reacher", []),
    ],
    "smoke": [
        ("claim_a", "claim_a", ["--cases", "300:nominal", "--num-seeds", "3", "--eval-steps", "3", "--sigmas", "1.0", "2.0"]),
        ("crossover_sweep", "crossover_sweep", ["--k-values", "100", "300", "--num-seeds", "2", "--eval-steps", "2"]),
        ("mujoco_smoke", "mujoco_smoke", []),
    ],
}


def run_suite(name, tag="local", out_dir=None, only=None, skip=None):
    if name not in SUITES:
        raise SystemExit(f"unknown suite {name!r}; choose from {', '.join(SUITES)}")
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    suite_dir = Path(out_dir) if out_dir else RESULTS_ROOT / tag / f"{name}-{stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    print(f"[suite] {name} -> {suite_dir}")
    failures = []
    for label, experiment, extra in SUITES[name]:
        if (only and label not in only) or (skip and label in skip):
            continue
        out = suite_dir / label
        if (out / "DONE").exists():
            print(f"[suite] {label}: already done, skipping")
            continue
        argv = [a.format(suite_dir=suite_dir) for a in extra]
        command = [sys.executable, "-m", "ua_mbrl", "run", experiment, "--out-dir", str(out), *argv]
        print(f"[suite] {label}: {' '.join(command[2:])}", flush=True)
        start = time.perf_counter()
        # Each experiment in a fresh process: no CUDA allocator state carries over.
        result = subprocess.run(command)
        print(f"[suite] {label}: exit {result.returncode} after {time.perf_counter() - start:.0f}s", flush=True)
        if result.returncode != 0:
            failures.append(label)
    (suite_dir / "SUITE_DONE").write_text("\n".join(failures))
    if failures:
        print(f"[suite] failed: {', '.join(failures)}")
    return suite_dir, failures
