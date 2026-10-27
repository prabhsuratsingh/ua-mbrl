"""Run experiments on vast.ai GPUs with one command.

    uv run ua-mbrl remote run --gpu RTX_4090 --suite replicate
    uv run ua-mbrl remote run --gpu RTX_4090 H100_SXM --suite latency      # GPUs in parallel
    uv run ua-mbrl remote run --gpu RTX_4090 --experiment claim_a -- --num-seeds 100

For each GPU this: rents the cheapest matching verified offer, attaches your
SSH key, ships the code (clones the exact local commit from GitHub, or
uploads the working tree), installs it with ``uv sync --frozen``, runs the
suite/experiment detached on the instance (so a dropped SSH connection does
not kill it), streams its log, copies the results back to
``results/vast/<gpu>/<run-id>/``, and destroys the instance -- also on
failure or Ctrl-C, after fetching whatever was produced (``--keep`` to skip).

Credentials come from ``.env`` in the repository root (see .env.example):
VAST_API_KEY, VAST_SSH_KEY (private key path; <path>.pub must exist),
GITHUB_TOKEN + GITHUB_REPO (owner/name; only for --source github).
They are passed to child processes via environment/stdin, never on a
command line or in logs.

Other commands: ``offers`` (list matching offers and prices), ``instances``,
``fetch`` (copy results from a kept instance), ``destroy``.
"""

import argparse
import concurrent.futures
import datetime as _dt
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from ua_mbrl.utils.results import REPO_ROOT

DEFAULT_IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
REMOTE_ROOT = "/workspace/ua-mbrl"
LOCAL_ROOT = REPO_ROOT / "results" / "vast"
_print_lock = threading.Lock()


# --- configuration ----------------------------------------------------------

def load_env(path=REPO_ROOT / ".env"):
    """Minimal .env parser (KEY=VALUE, # comments, optional quotes).
    Values already set in the process environment win."""
    values = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key in list(values):
        if os.environ.get(key):
            values[key] = os.environ[key]
    for key in ("VAST_API_KEY", "VAST_SSH_KEY", "GITHUB_TOKEN", "GITHUB_REPO", "VAST_IMAGE"):
        if key not in values and os.environ.get(key):
            values[key] = os.environ[key]
    return values


class Config:
    def __init__(self, env):
        self.env = env
        self.api_key = env.get("VAST_API_KEY", "")
        self.ssh_key = Path(os.path.expanduser(env.get("VAST_SSH_KEY", "~/.ssh/id_ed25519")))
        self.github_token = env.get("GITHUB_TOKEN", "")
        self.github_repo = env.get("GITHUB_REPO", "")
        self.image = env.get("VAST_IMAGE", DEFAULT_IMAGE)
        self.secrets = [s for s in (self.api_key, self.github_token) if s]

    def require_vast(self):
        missing = []
        if not self.api_key:
            missing.append("VAST_API_KEY")
        if not self.ssh_key.exists():
            missing.append(f"VAST_SSH_KEY (no private key at {self.ssh_key})")
        elif not Path(str(self.ssh_key) + ".pub").exists():
            missing.append(f"public key {self.ssh_key}.pub")
        if missing:
            raise SystemExit("missing in .env: " + ", ".join(missing) + "  (see .env.example)")

    def redact(self, text):
        for secret in self.secrets:
            text = text.replace(secret, "***")
        return text


def log(prefix, message):
    with _print_lock:
        print(f"[{prefix}] {message}", flush=True)


# --- vast CLI wrapper -------------------------------------------------------

# The vastai CLI runs in its own uvx environment: it cannot share the project's
# lock (the PyTorch index's old `requests` shadows the version vastai needs).
VASTAI_VERSION = "1.9.1"


def _vastai_command():
    if shutil.which("uvx"):
        return ["uvx", "--from", f"vastai=={VASTAI_VERSION}", "vastai"]
    exe = shutil.which("vastai")
    if exe:
        return [exe]
    raise SystemExit("needs `uvx` (ships with uv) or a `vastai` CLI on PATH")


def vast(config, *args, raw=True, check=True):
    command = _vastai_command() + list(args) + (["--raw"] if raw else [])
    env = {**os.environ, "VAST_API_KEY": config.api_key, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(command, capture_output=True, text=True, env=env, encoding="utf-8")
    output = (result.stdout or "").strip()
    if check and result.returncode != 0:
        raise RuntimeError(config.redact(f"vastai {' '.join(args)} failed: {result.stderr or output}"))
    if not raw:
        return output
    try:
        return json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(config.redact(f"vastai {' '.join(args)}: unexpected output {output[:500]!r}")) from error


def search_offers(config, gpu, disk_gb, max_dph=None, exclusive=False, min_reliability=0.98, limit=10):
    query = (
        f"gpu_name={gpu} num_gpus=1 rentable=true verified=true reliability>{min_reliability} "
        f"disk_space>={disk_gb} cuda_vers>=12.8 direct_port_count>=1"
    )
    if exclusive:
        query += " gpu_frac=1"  # the whole machine is ours: no co-tenants on its CPU/PCIe
    if max_dph:
        query += f" dph<={max_dph}"
    offers = vast(config, "search", "offers", query, "-o", "dph", "--limit", str(limit), "--storage", str(disk_gb))
    return offers if isinstance(offers, list) else offers.get("offers", [])


def describe_offer(offer):
    return (
        f"offer {offer.get('id')}: {offer.get('gpu_name')} {offer.get('gpu_ram', 0) / 1024:.0f}GB, "
        f"${offer.get('dph_total', 0):.3f}/h, reliability {offer.get('reliability2', offer.get('reliability', 0)):.3f}, "
        f"CUDA {offer.get('cuda_max_good')}, driver {offer.get('driver_version')}, "
        f"{offer.get('cpu_name', '?')} ({offer.get('cpu_cores_effective', '?')} cores), "
        f"{offer.get('geolocation', '?')}"
    )


def show_instance(config, instance_id):
    data = vast(config, "show", "instance", str(instance_id))
    return data.get("instances", data) if isinstance(data, dict) else data


def destroy_instance(config, instance_id, prefix):
    try:
        vast(config, "destroy", "instance", str(instance_id), raw=False)
        log(prefix, f"instance {instance_id} destroyed")
    except RuntimeError as error:
        log(prefix, f"!! could not destroy instance {instance_id}: {error} -- destroy it at https://cloud.vast.ai/instances/")


# --- ssh helpers ------------------------------------------------------------

class Remote:
    def __init__(self, config, host, port, prefix):
        self.config = config
        self.host = host
        self.port = port
        self.prefix = prefix
        known_hosts = LOCAL_ROOT / ".known_hosts"
        known_hosts.parent.mkdir(parents=True, exist_ok=True)
        self.options = [
            "-i", str(config.ssh_key),
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={known_hosts}",
            "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6",
            "-o", "LogLevel=ERROR",
        ]

    def ssh(self, command, script=None, check=True, capture=True, timeout=None):
        argv = ["ssh", *self.options, "-p", str(self.port), f"root@{self.host}", command]
        result = subprocess.run(
            argv, input=script, capture_output=capture, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
        if check and result.returncode != 0:
            detail = (result.stderr or "")[-2000:] if capture else ""
            raise RuntimeError(self.config.redact(f"ssh `{command}` exited {result.returncode}: {detail}"))
        return result

    def upload(self, local_path, remote_path):
        subprocess.run(["scp", *self.options, "-P", str(self.port), str(local_path), f"root@{self.host}:{remote_path}"],
                       check=True, capture_output=True)

    def download(self, remote_path, local_path):
        subprocess.run(["scp", *self.options, "-P", str(self.port), f"root@{self.host}:{remote_path}", str(local_path)],
                       check=True, capture_output=True)

    def wait_ready(self, timeout_s=900):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                if self.ssh("true", check=False, timeout=60).returncode == 0:
                    return
            except subprocess.TimeoutExpired:
                pass
            time.sleep(10)
        raise RuntimeError(f"ssh to {self.host}:{self.port} not ready after {timeout_s}s")


def ssh_endpoint(config, instance_id):
    url = vast(config, "ssh-url", str(instance_id), raw=False).strip()
    # ssh://root@HOST:PORT
    if not url.startswith("ssh://"):
        return None
    host_port = url.split("@", 1)[1]
    host, port = host_port.rsplit(":", 1)
    return host, int(port)


# --- source -----------------------------------------------------------------

def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def commit_on_github(config, commit):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{config.github_repo}/commits/{commit}",
        headers={"Authorization": f"Bearer {config.github_token}", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status == 200
    except urllib.error.HTTPError:
        return False


def resolve_source(config, requested):
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    if requested in ("github", "auto") and config.github_repo and config.github_token:
        pushed = commit_on_github(config, commit)
        if pushed and not dirty:
            return "github", commit
        if requested == "github":
            raise SystemExit(
                f"--source github needs a clean tree whose HEAD ({commit[:10]}) is pushed to {config.github_repo} "
                f"(dirty={dirty}, pushed={pushed}). Commit and push, or use --source upload."
            )
        print(f"[source] HEAD {commit[:10]} {'has uncommitted changes' if dirty else 'is not on GitHub'}; "
              f"uploading the working tree instead")
    elif requested == "github":
        raise SystemExit("--source github needs GITHUB_TOKEN and GITHUB_REPO in .env")
    return "upload", commit + ("-dirty" if dirty else "")


def working_tree_tarball():
    """Tracked + untracked-but-not-ignored files (what `git status` sees), minus
    local results other than the published reference data."""
    files = _git("ls-files", "-c", "-o", "--exclude-standard").splitlines()
    keep = [f for f in files if not f.startswith("results/") or f.startswith("results/published/")]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in keep:
            path = REPO_ROOT / name
            if path.is_file():
                tar.add(path, arcname=name)
    handle = tempfile.NamedTemporaryFile(suffix=".tgz", delete=False)
    handle.write(buffer.getvalue())
    handle.close()
    return Path(handle.name)


# --- remote scripts -------------------------------------------------------------

def bootstrap_script(config, source, commit):
    lines = [
        "set -euo pipefail",
        "export DEBIAN_FRONTEND=noninteractive",
        "if ! command -v git >/dev/null || ! command -v curl >/dev/null; then",
        "  apt-get update -qq >/dev/null && apt-get install -y -qq git curl ca-certificates >/dev/null; fi",
        "if ! command -v uv >/dev/null && [ ! -x \"$HOME/.local/bin/uv\" ]; then",
        "  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; fi",
        "export PATH=\"$HOME/.local/bin:$PATH\"",
        f"rm -rf {REMOTE_ROOT} && mkdir -p {REMOTE_ROOT}",
    ]
    if source == "github":
        lines += [
            f"git clone -q https://x-access-token:{config.github_token}@github.com/{config.github_repo}.git {REMOTE_ROOT}",
            f"cd {REMOTE_ROOT} && git checkout -q {commit}",
            f"git remote set-url origin https://github.com/{config.github_repo}.git",
        ]
    else:
        lines += [f"tar -xzf /workspace/ua-mbrl-src.tgz -C {REMOTE_ROOT}", f"cd {REMOTE_ROOT}"]
    lines += [
        "uv sync --frozen --quiet",
        "uv run --frozen ua-mbrl env",
        "nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv",
    ]
    return "\n".join(lines) + "\n"


def launch_command(run_id, commit, ua_args):
    out = f"{REMOTE_ROOT}/results/remote/{run_id}"
    inner = (
        f"cd {REMOTE_ROOT} && export PATH=$HOME/.local/bin:$PATH UA_MBRL_COMMIT={commit} && "
        f"mkdir -p {out} && (uv run --frozen ua-mbrl {' '.join(shlex.quote(a) for a in ua_args)}; "
        f"echo $? > {out}/EXIT_CODE) > {out}/remote.log 2>&1"
    )
    return f"nohup bash -c {shlex.quote(inner)} > /dev/null 2>&1 &", out


# --- one GPU, end to end --------------------------------------------------------

def run_on_gpu(config, gpu, args, source, commit, tarball):
    prefix = gpu
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{args.suite or args.experiment}-{stamp}"
    local_dir = LOCAL_ROOT / gpu.lower() / run_id
    if args.suite:
        ua_args = ["suite", args.suite, "--tag", gpu, "--out-dir", f"results/remote/{run_id}"]
        ua_args += (["--only", *args.only] if args.only else []) + (["--skip", *args.skip] if args.skip else [])
    else:
        ua_args = ["run", args.experiment, "--out-dir", f"results/remote/{run_id}/{args.experiment}", *args.experiment_args]

    if args.offer:
        offers = [o for o in search_offers(config, gpu, args.disk, None, False, 0.0, 200) if o.get("id") == args.offer]
    else:
        offers = search_offers(config, gpu, args.disk, args.max_dph, args.exclusive)
    if not offers:
        raise RuntimeError(f"no rentable verified {gpu} offers matching the filters (try --max-dph / no --exclusive)")
    offer = offers[0]
    log(prefix, "renting " + describe_offer(offer))
    created = vast(config, "create", "instance", str(offer["id"]), "--image", config.image,
                   "--disk", str(args.disk), "--ssh", "--direct", "--label", f"ua-mbrl-{run_id}", "--cancel-unavail")
    instance_id = created.get("new_contract")
    if not instance_id:
        raise RuntimeError(f"instance creation failed: {created}")
    log(prefix, f"instance {instance_id} created (destroy manually at https://cloud.vast.ai/instances/ if this process dies)")
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "vast_offer.json").write_text(json.dumps(offer, indent=2, default=str))

    started = time.time()
    remote = None
    try:
        pubkey = Path(str(config.ssh_key) + ".pub").read_text().strip()
        attached = vast(config, "attach", "ssh", str(instance_id), pubkey, raw=False, check=False)
        log(prefix, f"attach ssh key: {attached[:200] or 'ok'}")

        deadline = time.time() + args.boot_timeout
        status = None
        while time.time() < deadline:
            info = show_instance(config, instance_id) or {}
            if info.get("actual_status") != status:
                status = info.get("actual_status")
                log(prefix, f"status: {status} {info.get('status_msg') or ''}".strip())
            if status == "running":
                break
            time.sleep(15)
        else:
            raise RuntimeError(f"instance not running after {args.boot_timeout}s")

        endpoint = None
        while endpoint is None and time.time() < deadline + 300:
            endpoint = ssh_endpoint(config, instance_id)
            if endpoint is None:
                time.sleep(10)
        if endpoint is None:
            raise RuntimeError("no ssh endpoint for instance")
        remote = Remote(config, *endpoint, prefix)
        log(prefix, f"waiting for ssh root@{endpoint[0]}:{endpoint[1]}")
        remote.wait_ready()

        if source == "upload":
            remote.upload(tarball, "/workspace/ua-mbrl-src.tgz")
        log(prefix, f"installing {source} {commit[:12]} (uv sync)")
        result = remote.ssh("bash -s", script=bootstrap_script(config, source, commit.replace("-dirty", "")), timeout=3600)
        (local_dir / "bootstrap.log").write_text(config.redact(result.stdout + result.stderr), encoding="utf-8")
        log(prefix, "environment ready; " + " ".join(result.stdout.strip().splitlines()[-1:]))

        command, remote_out = launch_command(run_id, commit, ua_args)
        log(prefix, f"launching: ua-mbrl {' '.join(ua_args)}")
        remote.ssh(command)
        stream_until_done(remote, remote_out, prefix, args.max_hours, started)
    except KeyboardInterrupt:
        log(prefix, "interrupted")
        raise
    except Exception as error:  # noqa: BLE001 -- report, fetch what exists, clean up
        log(prefix, f"!! {config.redact(str(error))}")
    finally:
        if remote is not None:
            try:
                fetch_results(remote, run_id, local_dir, prefix)
            except Exception as error:  # noqa: BLE001
                log(prefix, f"!! fetch failed: {error}")
        hours = (time.time() - started) / 3600
        cost = hours * float(offer.get("dph_total") or 0)
        (local_dir / "vast_run.json").write_text(json.dumps({
            "gpu": gpu, "instance_id": instance_id, "offer_id": offer.get("id"), "run_id": run_id,
            "commit": commit, "source": source, "ua_args": ua_args, "hours": round(hours, 3),
            "approx_cost_usd": round(cost, 2), "kept": args.keep,
        }, indent=2))
        if args.keep:
            log(prefix, f"--keep: instance {instance_id} left running (destroy: uv run ua-mbrl remote destroy {instance_id})")
        else:
            destroy_instance(config, instance_id, prefix)
        log(prefix, f"done in {hours:.2f} h, ~${cost:.2f}; results in {local_dir}")
    return local_dir


def stream_until_done(remote, remote_out, prefix, max_hours, started):
    offset = 0
    while True:
        result = remote.ssh(f"tail -c +{offset + 1} {remote_out}/remote.log 2>/dev/null; true", check=False, timeout=120)
        chunk = result.stdout or ""
        if chunk:
            offset += len(chunk.encode("utf-8"))
            for line in chunk.splitlines():
                log(prefix, line)
        done = remote.ssh(f"cat {remote_out}/EXIT_CODE 2>/dev/null", check=False, timeout=60)
        if done.returncode == 0 and done.stdout.strip():
            log(prefix, f"remote run finished with exit code {done.stdout.strip()}")
            return
        if time.time() - started > max_hours * 3600:
            raise RuntimeError(f"--max-hours {max_hours} exceeded; fetching partial results")
        time.sleep(30)


def fetch_results(remote, run_id, local_dir, prefix):
    archive = f"/workspace/{run_id}.tgz"
    remote.ssh(f"cd {REMOTE_ROOT}/results/remote && tar -czf {archive} {run_id}", timeout=900)
    local_archive = local_dir / f"{run_id}.tgz"
    remote.download(archive, local_archive)
    with tarfile.open(local_archive) as tar:
        tar.extractall(local_dir.parent, filter="data")
    local_archive.unlink()
    log(prefix, f"results fetched to {local_dir}")


# --- commands -------------------------------------------------------------------

def cmd_run(config, args):
    config.require_vast()
    if bool(args.suite) == bool(args.experiment):
        raise SystemExit("give exactly one of --suite or --experiment")
    source, commit = resolve_source(config, args.source)
    tarball = working_tree_tarball() if source == "upload" else None
    print(f"[source] {source} @ {commit[:12]}; GPUs: {', '.join(args.gpu)}; image {config.image}")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpu)) as pool:
            futures = {pool.submit(run_on_gpu, config, gpu, args, source, commit, tarball): gpu for gpu in args.gpu}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001
                    log(futures[future], f"!! {config.redact(str(error))}")
    finally:
        if tarball:
            tarball.unlink(missing_ok=True)
    return 0


def cmd_offers(config, args):
    # Offer search works without an API key.
    for gpu in args.gpu:
        offers = search_offers(config, gpu, args.disk, args.max_dph, args.exclusive, limit=args.limit)
        print(f"\n{gpu}: {len(offers)} offer(s)")
        for offer in offers:
            print("  " + describe_offer(offer))
    return 0


def cmd_instances(config, args):
    for row in vast(config, "show", "instances"):
        print(f"{row.get('id')}  {row.get('actual_status')}  {row.get('gpu_name')}  ${row.get('dph_total', 0):.3f}/h  "
              f"{row.get('label') or ''}")
    return 0


def cmd_destroy(config, args):
    for instance_id in args.instance_ids:
        destroy_instance(config, instance_id, "destroy")
    return 0


def cmd_fetch(config, args):
    config.require_vast()
    endpoint = ssh_endpoint(config, args.instance_id)
    if endpoint is None:
        raise SystemExit("no ssh endpoint for that instance")
    remote = Remote(config, *endpoint, "fetch")
    listing = remote.ssh(f"ls {REMOTE_ROOT}/results/remote", check=False).stdout.split()
    run_ids = [args.run_id] if args.run_id else [r for r in listing if not r.endswith(".tgz")]
    for run_id in run_ids:
        fetch_results(remote, run_id, LOCAL_ROOT / args.gpu.lower() / run_id, "fetch")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="ua-mbrl remote", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def gpu_filters(p):
        p.add_argument("--gpu", nargs="+", required=True, help="vast gpu_name(s), e.g. RTX_4090 RTX_5090 H100_SXM A100_SXM4")
        p.add_argument("--disk", type=int, default=40, help="GB")
        p.add_argument("--max-dph", type=float, default=None, help="max $/hour")
        p.add_argument("--exclusive", action="store_true", help="only offers renting the whole machine (latency runs)")

    run = commands.add_parser("run", help="rent, run, fetch results, destroy")
    gpu_filters(run)
    run.add_argument("--suite", default=None, help="suite name from ua_mbrl.suites (e.g. replicate, latency)")
    run.add_argument("--only", nargs="+", default=None, help="suite labels to run")
    run.add_argument("--skip", nargs="+", default=None, help="suite labels to skip")
    run.add_argument("--experiment", default=None, help="single experiment; its options go after --")
    run.add_argument("--source", choices=("auto", "github", "upload"), default="auto",
                     help="auto: clone from GitHub if HEAD is pushed and the tree is clean, else upload the working tree")
    run.add_argument("--offer", type=int, default=None, help="rent this specific offer id")
    run.add_argument("--keep", action="store_true", help="do not destroy the instance afterwards")
    run.add_argument("--max-hours", type=float, default=12.0, help="stop waiting (fetch partial results, destroy) after this")
    run.add_argument("--boot-timeout", type=int, default=1200, help="seconds to wait for the instance to start")
    run.add_argument("experiment_args", nargs=argparse.REMAINDER)

    offers = commands.add_parser("offers", help="list matching offers")
    gpu_filters(offers)
    offers.add_argument("--limit", type=int, default=5)

    commands.add_parser("instances", help="list your instances")

    destroy = commands.add_parser("destroy", help="destroy instance(s)")
    destroy.add_argument("instance_ids", nargs="+")

    fetch = commands.add_parser("fetch", help="copy results from a running (kept) instance")
    fetch.add_argument("instance_id")
    fetch.add_argument("--gpu", required=True, help="local folder name under results/vast/")
    fetch.add_argument("--run-id", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "experiment_args", None) and args.experiment_args[:1] == ["--"]:
        args.experiment_args = args.experiment_args[1:]
    config = Config(load_env())
    handlers = {"run": cmd_run, "offers": cmd_offers, "instances": cmd_instances,
                "destroy": cmd_destroy, "fetch": cmd_fetch}
    return handlers[args.command](config, args)


if __name__ == "__main__":
    sys.exit(main())
