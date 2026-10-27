import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

LEGACY_DIR = Path(__file__).resolve().parents[1] / "legacy"


@pytest.fixture(scope="session")
def legacy():
    """Import a frozen legacy script by module name, e.g. legacy("claim_a_extended")."""
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))
    return importlib.import_module


@pytest.fixture
def run_dir(tmp_path):
    from ua_mbrl.utils.results import RunDir

    return RunDir(tmp_path / "run")


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def assert_same(a, b, path="result"):
    """Recursive exact equality for dicts/lists/arrays/floats."""
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys differ {set(a) ^ set(b)}"
        for key in a:
            assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        assert a_arr.shape == b_arr.shape, f"{path}: shape {a_arr.shape} != {b_arr.shape}"
        assert np.array_equal(a_arr, b_arr), f"{path}: {a_arr} != {b_arr}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"
