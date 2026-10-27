"""Command-line entry point.

    ua-mbrl list                          # experiments and the paper results they produce
    ua-mbrl run <experiment> [options]    # results/<tag>/<experiment>/<timestamp>/
    ua-mbrl suite <name> [--tag TAG]      # a named list of runs (e.g. replicate), then check
    ua-mbrl check <suite-dir>             # compare a suite run with the published numbers
    ua-mbrl figures [--out-dir DIR]       # paper figures
    ua-mbrl env                           # GPU / driver / library versions
    ua-mbrl remote <command> [options]    # vast.ai runs (see ua_mbrl.remote.vast)
"""

import argparse
import json
import sys
import time

from ua_mbrl.experiments import REGISTRY


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _build_parser():
    parser = argparse.ArgumentParser(prog="ua-mbrl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="list experiments")
    commands.add_parser("env", help="print environment info as JSON")

    figures = commands.add_parser("figures", help="regenerate paper figures")
    figures.add_argument("--out-dir", default=None)

    suite = commands.add_parser("suite", help="run a named list of experiments (ua_mbrl.suites), then check it")
    suite.add_argument("name")
    suite.add_argument("--tag", default="local", help="results/<tag>/..., e.g. a GPU name")
    suite.add_argument("--out-dir", default=None, help="resume/write into this suite directory")
    suite.add_argument("--only", nargs="+", default=None, help="run only these labels")
    suite.add_argument("--skip", nargs="+", default=None, help="skip these labels")
    suite.add_argument("--no-check", action="store_true")

    check = commands.add_parser("check", help="compare a suite directory with the published results")
    check.add_argument("suite_dir")

    remote = commands.add_parser("remote", help="run on vast.ai (ua-mbrl remote --help)", add_help=False)
    remote.add_argument("remote_args", nargs=argparse.REMAINDER)

    run = commands.add_parser("run", help="run an experiment")
    experiments = run.add_subparsers(dest="experiment", required=True)
    for name, experiment in REGISTRY.items():
        sub = experiments.add_parser(name, help=experiment.summary, description=f"{experiment.summary}\nPaper: {experiment.paper}")
        sub.add_argument("--out-dir", default=None, help="write results here instead of results/<tag>/<name>/<timestamp>")
        sub.add_argument("--tag", default="local", help="results/<tag>/... grouping, e.g. a GPU name")
        sub.add_argument(
            "--pilot-rng", choices=("isolated", "shared"), default="isolated",
            help="adaptive pilot noise stream: isolated (current) or shared (pre-fix controller, "
                 "needed to reproduce results published before the fix)",
        )
        add_arguments, _ = experiment.load()
        add_arguments(sub)
    return parser


def run_experiment(name, args):
    from ua_mbrl.utils.results import RunDir

    from ua_mbrl.core.mppi import set_default_pilot_rng

    _, run_fn = REGISTRY[name].load()
    set_default_pilot_rng(getattr(args, "pilot_rng", "isolated"))
    run_dir = RunDir.create(name, out_dir=args.out_dir, tag=args.tag)
    experiment_args = {k: v for k, v in vars(args).items() if k not in ("command", "experiment")}
    run_dir.write_meta(name, experiment_args)
    log = open(run_dir.path / "stdout.log", "a", encoding="utf-8")
    stdout = sys.stdout
    sys.stdout = _Tee(stdout, log)
    start = time.perf_counter()
    try:
        print(f"[ua-mbrl] {name} -> {run_dir.path}")
        result = run_fn(args, run_dir)
        print(f"[ua-mbrl] {name} finished in {time.perf_counter() - start:.1f}s -> {run_dir.path}")
        (run_dir.path / "DONE").write_text(f"{time.perf_counter() - start:.1f}\n")
        return result
    finally:
        sys.stdout = stdout
        log.close()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "remote":
        from ua_mbrl.remote.vast import main as remote_main
        return remote_main(argv[1:])

    args = _build_parser().parse_args(argv)
    if args.command == "list":
        width = max(len(n) for n in REGISTRY)
        for name, experiment in REGISTRY.items():
            print(f"{name:<{width}}  {experiment.summary}\n{'':<{width}}  paper: {experiment.paper}")
    elif args.command == "env":
        from ua_mbrl.utils.results import environment_info
        print(json.dumps(environment_info(), indent=2))
    elif args.command == "figures":
        from ua_mbrl.figures import generate_all
        generate_all(args.out_dir)
    elif args.command == "run":
        run_experiment(args.experiment, args)
    elif args.command == "suite":
        from ua_mbrl.replication import check_suite
        from ua_mbrl.suites import run_suite
        suite_dir, failures = run_suite(args.name, tag=args.tag, out_dir=args.out_dir, only=args.only, skip=args.skip)
        if not args.no_check:
            check_suite(suite_dir)
        return 1 if failures else 0
    elif args.command == "check":
        from ua_mbrl.replication import check_suite
        check_suite(args.suite_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
