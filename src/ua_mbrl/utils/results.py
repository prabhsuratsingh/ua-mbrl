"""Run directories, JSONL records and environment metadata.

Every experiment run writes into its own directory::

    results/<tag>/<experiment>/<timestamp>/
        meta.json        # args, git commit, GPU/driver/torch versions
        records.jsonl    # one JSON object per completed unit of work
"""

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = Path(os.environ.get("UA_MBRL_RESULTS", REPO_ROOT / "results"))
PUBLISHED_DIR = REPO_ROOT / "results" / "published"


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (Path, torch.device)):
        return str(value)
    return value


def _git(*args):
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "git_commit": _git("rev-parse", "HEAD") or os.environ.get("UA_MBRL_COMMIT", ""),
        "git_dirty": bool(_git("status", "--porcelain")),
        "hostname": platform.node(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_memory_mb"] = props.total_memory // (1024 * 1024)
        info["gpu_capability"] = f"{props.major}.{props.minor}"
        try:
            info["nvidia_driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            info["nvidia_driver"] = ""
    try:
        import mujoco
        info["mujoco"] = mujoco.__version__
    except ImportError:
        pass
    return info


class RunDir:
    """One experiment run's output directory."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.records_path = self.path / "records.jsonl"

    @classmethod
    def create(cls, experiment, out_dir=None, tag="local"):
        if out_dir is not None:
            return cls(Path(out_dir))
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(RESULTS_ROOT / tag / experiment / stamp)

    def write_meta(self, experiment, args):
        meta = {
            "experiment": experiment,
            "args": to_jsonable(args),
            "started": _dt.datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv,
            "environment": environment_info(),
        }
        (self.path / "meta.json").write_text(json.dumps(meta, indent=2))

    def append(self, record):
        with open(self.records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(record)) + "\n")

    def records(self):
        return read_jsonl(self.records_path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
