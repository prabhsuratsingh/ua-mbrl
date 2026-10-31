"""Offline tests for the vast.ai runner (no network, no instances rented)."""

import json
import tarfile

import pytest

from ua_mbrl.remote import vast


def test_load_env_parses_and_environment_wins(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nVAST_API_KEY="abc"\nGITHUB_REPO=me/repo\n\nVAST_SSH_KEY=~/.ssh/k\n')
    monkeypatch.setenv("GITHUB_REPO", "other/repo")
    values = vast.load_env(env_file)
    assert values["VAST_API_KEY"] == "abc"
    assert values["GITHUB_REPO"] == "other/repo"
    assert values["VAST_SSH_KEY"] == "~/.ssh/k"


def test_redact_hides_secrets():
    config = vast.Config({"VAST_API_KEY": "sekrit-key", "GITHUB_TOKEN": "ghp_token"})
    assert config.redact("x sekrit-key y ghp_token") == "x *** y ***"


def test_bootstrap_script_github_vs_upload():
    config = vast.Config({"GITHUB_TOKEN": "ghp_token", "GITHUB_REPO": "me/repo"})
    github = vast.bootstrap_script(config, "github", "abc123")
    assert "x-access-token:ghp_token@github.com/me/repo.git" in github
    assert "git checkout -q abc123" in github
    assert "set-url origin https://github.com/me/repo.git" in github  # token not left in .git/config
    upload = vast.bootstrap_script(config, "upload", "abc123")
    assert "ghp_token" not in upload
    assert "uv sync --frozen" in upload


def test_launch_command_is_detached_and_records_exit_code():
    command, out = vast.launch_command("run-1", "abc", ["run", "claim_a", "--cases", "300:nominal"])
    assert command.startswith("nohup bash -c ")
    assert command.endswith("&")
    assert "EXIT_CODE" in command and out.endswith("results/remote/run-1")
    assert "UA_MBRL_COMMIT=abc" in command


def test_working_tree_tarball_contents():
    path = vast.working_tree_tarball()
    try:
        with tarfile.open(path) as tar:
            names = set(tar.getnames())
    finally:
        path.unlink()
    assert "pyproject.toml" in names and "uv.lock" in names
    assert "src/ua_mbrl/cli.py" in names
    assert any(n.startswith("results/published/") for n in names)
    assert not any(n.startswith("results/local") or n.startswith("results/vast") for n in names)
    assert ".env" not in names


def test_cli_parses_experiment_passthrough():
    args = vast.build_parser().parse_args(
        ["run", "--gpu", "RTX_4090", "H100_SXM", "--experiment", "claim_a", "--", "--num-seeds", "5"]
    )
    assert args.gpu == ["RTX_4090", "H100_SXM"]
    assert args.experiment_args == ["--", "--num-seeds", "5"]


class FakeRemote:
    calls = []

    def __init__(self, config, host, port, prefix):
        self.host, self.port = host, port

    def wait_ready(self, timeout_s=900):
        FakeRemote.calls.append("wait_ready")

    def upload(self, local, remote):
        FakeRemote.calls.append(("upload", remote))

    def ssh(self, command, script=None, check=True, capture=True, timeout=None):
        FakeRemote.calls.append(("ssh", command.split()[0]))

        class Result:
            returncode = 0
            stdout = "0\n" if command.startswith("cat ") else "env ok\n"
            stderr = ""
        if FakeRemote.fail_on_launch and command.startswith("nohup"):
            raise RuntimeError("launch failed")
        return Result()


@pytest.mark.parametrize("fail", [False, True])
def test_run_on_gpu_flow_always_fetches_and_destroys(tmp_path, monkeypatch, fail):
    FakeRemote.calls = []
    FakeRemote.fail_on_launch = fail
    vast_calls = []

    def fake_vast(config, *args, raw=True, check=True):
        vast_calls.append(args[:2])
        if args[:2] == ("search", "offers"):
            return [{"id": 11, "gpu_name": "RTX 4090", "dph_total": 0.4, "gpu_ram": 24576}]
        if args[:2] == ("create", "instance"):
            return {"success": True, "new_contract": 99}
        if args[:2] == ("show", "instance"):
            return {"actual_status": "running"}
        if args[0] == "ssh-url":
            return "ssh://root@1.2.3.4:4567"
        return "" if not raw else {}

    fetched = []
    monkeypatch.setattr(vast, "vast", fake_vast)
    monkeypatch.setattr(vast, "Remote", FakeRemote)
    monkeypatch.setattr(vast, "fetch_results", lambda remote, run_id, local_dir, prefix: fetched.append(run_id))
    monkeypatch.setattr(vast, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(vast.time, "sleep", lambda s: None)

    key = tmp_path / "id_test"
    key.write_text("private")
    (tmp_path / "id_test.pub").write_text("ssh-ed25519 AAAA test")
    config = vast.Config({"VAST_API_KEY": "k", "VAST_SSH_KEY": str(key)})
    args = vast.build_parser().parse_args(["run", "--gpu", "RTX_4090", "--suite", "smoke"])

    local_dir = vast.run_on_gpu(config, "RTX_4090", args, "upload", "abc", tmp_path / "src.tgz")

    assert ("destroy", "instance") in vast_calls  # destroyed in both the success and failure case
    assert fetched, "results fetched even when the run fails"
    record = json.loads((local_dir / "vast_run.json").read_text())
    assert record["instance_id"] == 99 and record["ua_args"][:2] == ["suite", "smoke"]
    assert ("upload", "/workspace/ua-mbrl-src.tgz") in FakeRemote.calls


def test_ssh_sends_script_as_lf_bytes_and_no_console_stdin(monkeypatch):
    """Regression: on Windows, text-mode stdin turned \n into \r\n and bash
    failed with `set: pipefail\r: invalid option name`."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(kwargs)
        return vast.subprocess.CompletedProcess(argv, 0, b"out\n", b"")

    monkeypatch.setattr(vast.subprocess, "run", fake_run)
    remote = vast.Remote(vast.Config({}), "host", 22, "test")
    result = remote.ssh("bash -s", script="set -euo pipefail\necho hi\n")
    assert calls[0]["input"] == b"set -euo pipefail\necho hi\n"
    assert b"\r" not in calls[0]["input"]
    assert result.stdout == "out\n"
    remote.ssh("true")
    assert calls[1]["stdin"] is vast.subprocess.DEVNULL and calls[1]["input"] is None


def test_offers_exclude_countries(monkeypatch):
    queries = []

    def fake_vast(config, *args, raw=True, check=True):
        queries.append(args[2])
        return [{"id": 1, "geolocation": "Vietnam, VN"}, {"id": 2, "geolocation": "Germany, DE"},
                {"id": 3, "geolocation": "Jiangsu, CN"}, {"id": 4, "geolocation": None}]

    monkeypatch.setattr(vast, "vast", fake_vast)
    offers = vast.search_offers(vast.Config({}), "RTX_4090", 40)
    assert [o["id"] for o in offers] == [2, 4]
    assert "geolocation notin [CN,VN]" in queries[0]
    assert [o["id"] for o in vast.search_offers(vast.Config({}), "RTX_4090", 40, exclude_countries=[])] == [1, 2, 3, 4]
    assert "geolocation" not in queries[1]


def test_destroy_skips_confirmation_prompt(monkeypatch):
    seen = []
    monkeypatch.setattr(vast, "vast", lambda config, *args, raw=True, check=True: seen.append(args) or "")
    vast._ACTIVE[123] = "RTX_4090"
    assert vast.destroy_instance(vast.Config({}), 123, "t")
    assert seen == [("destroy", "instance", "123", "-y")]
    assert 123 not in vast._ACTIVE


def test_stop_event_interrupts_waits():
    vast._STOP.set()
    try:
        with pytest.raises(RuntimeError, match="Ctrl-C"):
            vast._sleep(30)
    finally:
        vast._STOP.clear()
