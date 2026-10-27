"""Every registered experiment runs end-to-end through the CLI at tiny scale
and writes meta.json / records.jsonl / stdout.log / DONE."""

import json

import pytest

from ua_mbrl.cli import main
from ua_mbrl.experiments import REGISTRY
from ua_mbrl.utils.results import PUBLISHED_DIR, read_jsonl

pytestmark = pytest.mark.slow

TINY = {
    "crossover_point": ["--k-max", "300", "--num-seeds", "3", "--eval-steps", "3"],
    "crossover_sweep": ["--k-values", "100", "300", "--num-seeds", "2", "--eval-steps", "2"],
    "confound_check": ["--k-values", "100", "--num-seeds", "2", "--eval-steps", "3"],
    "crossover_quality": ["--k-values", "100", "--num-seeds", "2", "--eval-steps", "3"],
    "claim_a": ["--cases", "300:nominal", "--num-seeds", "3", "--eval-steps", "3", "--sigmas", "1.0", "2.0"],
    "claim_a_stratified": ["--num-seeds", "2", "--eval-steps", "11"],
    "fallback": ["--num-seeds", "2", "--eval-steps", "11"],
    "ood_severity": ["--shift-levels", "0.0", "3.0", "--repeats", "2"],
    "ood_two_point": [],
    "robustness": ["--eval-steps", "3", "--episodes", "1"],
    "mujoco_regulation": ["--seeds", "7", "--steps", "3", "--episodes", "1"],
    "mujoco_claim_a": ["--seeds", "7", "--steps", "3", "--episodes", "1", "--sigmas", "0.2",
                       "--adaptive-results", str(PUBLISHED_DIR / "mujoco_quality_retuned_results.jsonl")],
    "reacher": ["--seeds", "7", "--steps", "3", "--episodes", "1"],
    "gpu_benchmark": ["--steps", "2", "--rollouts", "50", "--horizon", "3"],
    "mujoco_smoke": [],
}


def test_every_experiment_has_a_tiny_config():
    assert set(TINY) == set(REGISTRY)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_experiment_runs(name, tmp_path):
    out = tmp_path / name
    assert main(["run", name, "--out-dir", str(out), *TINY[name]]) == 0
    assert (out / "DONE").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["experiment"] == name
    assert "torch" in meta["environment"]
    assert read_jsonl(out / "records.jsonl"), "no records written"
    assert (out / "stdout.log").read_text()


def test_crossover_sweep_writes_analysis(tmp_path):
    out = tmp_path / "sweep"
    main(["run", "crossover_sweep", "--out-dir", str(out), "--k-values", "100", "300",
          "--num-seeds", "2", "--eval-steps", "2", "--mode", "inprocess"])
    analysis = json.loads((out / "analysis.json").read_text())
    assert analysis["k_values"] == [100, 300]


def test_figures(tmp_path):
    main(["figures", "--out-dir", str(tmp_path)])
    assert len(list(tmp_path.glob("fig*.png"))) == 5


def test_list_and_env(capsys):
    main(["list"])
    assert "claim_a" in capsys.readouterr().out
    main(["env"])
    assert "torch" in json.loads(capsys.readouterr().out)
