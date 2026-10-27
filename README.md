# ua-mbrl

GPU-batched, uncertainty-gated ensemble MPPI. This is the code behind
`arxiv_preprint.tex`. It locates the wall-clock crossover where adaptive
rollout allocation beats a fixed budget, and characterizes how reliable
ensemble disagreement is as an out-of-distribution (OOD) trigger.

## Setup

```powershell
uv sync                    # Python 3.12, torch cu128, mujoco, pytest
uv run ua-mbrl env         # GPU / driver / library versions
uv run ua-mbrl list        # every experiment and the paper result it produces
```

## Layout

```
src/ua_mbrl/
  core/          dynamics.py (residual MLP, bootstrap ensemble with stacked-einsum
                 inference/training), mppi.py (GenericMPPI, AdaptiveGenericMPPI,
                 SeedBatchedGenericMPPI), fallback.py, normalizer.py
  envs/          pendulum.py, two_link.py (MuJoCo regulation), reacher.py
  eval/          pendulum.py (shared seeded closed-loop runners), ood.py
  experiments/   one module per experiment, registered in experiments/__init__.py
  suites.py      named lists of runs (replicate, latency, smoke, ...)
  replication.py compares a suite run with the published numbers
  remote/vast.py one-command vast.ai runs
  figures.py     paper figures
  cli.py         `ua-mbrl` entry point
tests/           equivalence with legacy/, published-result checks, CLI, remote
legacy/          the original flat scripts, frozen (used by the equivalence tests)
results/
  published/     raw records behind the paper's MuJoCo and crossover tables
  local/ ...     your runs: results/<tag>/<experiment>/<timestamp>/
  vast/<gpu>/    runs fetched from vast.ai
```

## Running experiments

```powershell
uv run ua-mbrl run claim_a                                    # paper scale
uv run ua-mbrl run claim_a --cases 8000:nominal --num-seeds 100
uv run ua-mbrl run mujoco_regulation --sigma 0.2 --seeds 7 17 27
uv run ua-mbrl run crossover_sweep --k-values 1000 8000 16000 32000
```

Each run writes the following to `results/<tag>/<experiment>/<timestamp>/` (or to `--out-dir`):

- `meta.json`: arguments, git commit, GPU, driver, torch and CUDA versions
- `records.jsonl`: one record per completed unit, with per-seed outcomes
- `stdout.log`
- `DONE`

Run `uv run ua-mbrl run <experiment> --help` to see each experiment's options.

## Reproducing the paper

```powershell
uv run ua-mbrl suite replicate --tag local-gtx1650    # every paper result with code, full scale
uv run ua-mbrl check results/local-gtx1650/<suite-dir> # -> replication_report.md
```

`check` compares the run with the published numbers. Seeded results are
compared exactly: success rates, paired records, AUCs, and the per-seed MuJoCo
records. Latency is reported for information only, because it depends on the
hardware. The suite runs each experiment in a fresh process, and a stopped
suite resumes when you rerun it with the same `--out-dir`.

**Results from before the pilot-RNG fix.** `tab:confound`, `tab:mujoco`
(regulation and reaching) and the narrow-ensemble fallback rate were produced
with the controller as it was before that fix. At the time, the adaptive
pilot drew its noise from the shared global random stream. Running with
`--pilot-rng shared` restores exactly that behaviour, and the `replicate`
suite does so for those results. New experiments should keep the default,
`--pilot-rng isolated`.

**No script for these paper numbers:** `tab:sigmasweep`, the 33.95×
seed-batching row, the cached-inference and training rows of
`tab:gpuspeedups`, and the Claim A K=1000 nominal row.

## Tests

```powershell
uv run pytest                   # everything (about 5 min on a GTX 1650)
uv run pytest -m "not slow"     # fast: stats, published-result checks, remote runner
```

`tests/test_equivalence.py` replays each legacy script at reduced scale and
requires the package to reproduce its seeded outputs exactly. Those outputs
are thresholds, per-seed successes, effective K, ESS traces and AUCs. The test
fails if the refactor changed the order of random-number draws anywhere.

## vast.ai

1. Copy `.env.example` to `.env`. Set `VAST_API_KEY` and `VAST_SSH_KEY`.
   Optionally set `GITHUB_TOKEN` and `GITHUB_REPO`.
2. Nothing else to install: the runner calls a pinned `vastai` CLI through `uvx`.
3. Run:

```powershell
uv run ua-mbrl remote offers --gpu RTX_4090 RTX_5090 A100_SXM4           # prices, no key needed
uv run ua-mbrl remote run --gpu RTX_4090 --suite replicate
uv run ua-mbrl remote run --gpu RTX_4090 A100_SXM4 --suite latency --exclusive
uv run ua-mbrl remote run --gpu RTX_5090 --experiment claim_a -- --num-seeds 100
```

A single command does the whole run:

- **Rent:** picks the cheapest verified matching offer.
- **Ship code:** with `--source auto`, it clones your exact HEAD from GitHub if
  HEAD is pushed and the tree is clean. Otherwise it uploads the working tree.
- **Run:** installs with `uv sync --frozen`, runs detached on the instance and
  streams the log.
- **Fetch:** copies results to `results/vast/<gpu>/<run-id>/`. That folder also
  gets the offer details, timing and an approximate cost.
- **Destroy:** removes the instance. This also happens on failure or Ctrl-C,
  after fetching partial results.

Use `--keep` to leave the instance running. Then use `remote fetch` and
`remote destroy` to collect results and clean up.

`--exclusive` rents only whole machines. Use it for latency runs, so no
co-tenants share the CPU or PCIe.
