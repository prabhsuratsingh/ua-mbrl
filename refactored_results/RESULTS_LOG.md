# Results log: refactored codebase (`ua_mbrl` package)

Every number in this file was produced by the refactored `src/ua_mbrl`
package. None of it comes from the old flat scripts in `legacy/`, so numbers
here are not interchangeable with the numbers in `arxiv_preprint.tex` unless
an entry says they were checked against them.

## Conventions

- **New entries go at the bottom**, under a dated heading. Each entry gives:
  - the command that produced it;
  - the run directory (raw `records.jsonl`, `meta.json`, `stdout.log`);
  - the hardware and code version.
- **Number** means a measured value.
- **Observation** means an interpretation or a finding about the method.
  Each one says how strong the evidence is.
- **Controller version.** "isolated" is the default, the current controller.
  "shared" (`--pilot-rng shared`) is the pre-fix controller, where the adaptive
  pilot draws from the global random stream. Use "shared" only to reproduce
  results published before the fix.
- **95% intervals.**
  - "± t" is the Student-t interval over seeds (as in the paper's tables).
  - "± z" is the 1.96·SE interval (as in the experiments' printed output).

---

## 2026-10-26 / 27 · Refactor verification and full-scale replication (GTX 1650)

**Hardware and software:**
- GPU: NVIDIA GeForce GTX 1650 (4 GB), driver 555.97, laptop on AC power.
- Software: torch 2.11.0+cu128, cuDNN 9.19, MuJoCo 3.13.0, Python 3.12.9.
- Code: commit `7a291ef` plus the uncommitted refactor (`git_dirty: true` in
  every `meta.json`). From now on, commit before a run so that `meta.json`
  holds an exact hash.

**Commands:**
- `uv run ua-mbrl suite replicate --tag local-gtx1650 --out-dir results/local-gtx1650/replicate`
- `uv run ua-mbrl check results/local-gtx1650/replicate`

**Run directories:**
- `results/local-gtx1650/replicate/<experiment>/`: replication runs. The
  report is `replication_report.md` in the same folder.
- `results/local-gtx1650/current_controller/<experiment>/`: the same four
  experiments that were rerun in shared mode for the report (confound check,
  fallback, MuJoCo regulation σ=1.0, reacher), run once more with the
  current controller.

### Verification summary

- **Package vs. legacy scripts.** 64 tests pass (`uv run pytest`, 5 min).
  They include bit-exact equivalence with every legacy script at reduced
  scale. They also include a planted reordering of the random draws, which
  the tests caught.
- **Full scale vs. the paper.** 68 match, 0 mismatch, 17 information-only
  (latency).
- **What matches exactly:**
  - every seeded success rate, paired win/tie/loss record, sign-test p-value
    and AUC;
  - every per-seed MuJoCo and reacher record (threshold, success, effective
    K, final distance).

### Numbers: matched-sample-budget Claim A (pendulum, n=40, current controller)

| K_max | condition | ceiling σ / success | adaptive | eff. K | escalation rate | matched K, σ / fixed | fixed / adaptive / tie | p |
|---|---|---|---|---|---|---|---|---|
| 3000 | nominal | 3.0 / 100.0% | 82.5% | 299.3 | 2.56% | 299, 4.0 / 92.5% | 7 / 3 / 30 | 0.344 |
| 8000 | nominal | 3.0 / 100.0% | 82.5% | 402.8 | 2.25% | 403, 4.0 / 97.5% | 7 / 1 / 32 | 0.070 |
| 16000 | nominal | 4.0 / 100.0% | 82.5% | 682.1 | 2.88% | 682, 4.0 / 100.0% | 7 / 0 / 33 | 0.016 |
| 1000 | mild mismatch | 6.0 / 75.0% | 50.0% | 284.1 | 6.56% | 284, 6.0 / 75.0% | 14 / 4 / 22 | 0.031 |

Calibrated threshold: 0.008571 in all four cases. The escalation rate is the
fraction of control steps on which adaptive used K_max. It was not logged
before the refactor.

### Numbers: matched-ceiling quality (pendulum, n=20, 30 steps, current controller)

| K_max | retuned σ | fixed / adaptive at σ=1.0 | fixed / adaptive at retuned σ | paired, retuned (fixed / adaptive / tie) |
|---|---|---|---|---|
| 100 | 1.0 | 0.75 / 0.75 | 0.75 / 0.75 | 0 / 0 / 20 |
| 1000 | 5.0 | 0.40 / 0.70 | 0.90 / 0.75 | 4 / 1 / 15 |
| 8000 | 3.0 | 0.45 / 0.65 | 0.70 / 0.70 | 3 / 3 / 14 |

The retuned column equals `tab:confoundfixed`.

### Numbers: tab:confound (σ=1.0, n=20), both controller versions

| K_max | fixed | adaptive, shared (pre-fix) | adaptive, isolated (current) |
|---|---|---|---|
| 100 | 0.75 | 0.35 | 0.75 |
| 1000 | 0.40 | 0.35 | 0.70 |
| 8000 | 0.45 | 0.40 | 0.65 |

The shared column equals the published `tab:confound`.

### Numbers: MuJoCo regulation σ=1.0 (15 seeds × 3 episodes, ± t)

| method | shared (= published tab:mujoco) | isolated (current controller) |
|---|---|---|
| single | 42.2 ± 19.1% | 44.4 ± 16.6% |
| ensemble | 55.6 ± 16.6% | 55.6 ± 16.6% |
| adaptive | 62.2 ± 13.7% (eff. K 205.7 ± 33.8) | 55.6 ± 13.4% (eff. K 201.7 ± 29.9) |

Latency on this run, in ms per command: single 42.1, ensemble 56.0, adaptive
57.5 (isolated). The shared run measured 51.4, 65.4 and 66.9, which is a
different machine state. The paper reports 62.5, 80.1 and 83.9, so absolute
latency does not reproduce.

MuJoCo σ=0.2 (8 seeds) equals the published values exactly: single
45.8 ± 33.1%, ensemble 70.8 ± 23.3%, adaptive 70.8 ± 31.4%. MuJoCo matched
budget equals the published values exactly: fixed 83.3 ± 21.1% vs adaptive
70.8 ± 31.4%, record 3 / 5 / 0.

### Numbers: reacher (10 seeds × 6 episodes, ± t)

| method | shared (= published) | isolated (current controller) |
|---|---|---|
| single | 16.7 ± 11.2%, dist 0.344 ± 0.086 | 16.7 ± 11.2%, dist 0.348 ± 0.094 |
| ensemble | 33.3 ± 15.9%, dist 0.193 ± 0.030 | 35.0 ± 16.3%, dist 0.192 ± 0.032 |
| adaptive | 33.3 ± 14.9%, dist 0.208 ± 0.034 | 30.0 ± 11.0%, dist 0.212 ± 0.044 (eff. K 199.5) |

### Numbers: fallback (pendulum, 15 seeds, 40 steps)

| variant | method | shared: success / trigger % | isolated: success / trigger % |
|---|---|---|---|
| mismatch, nominal | adaptive / fallback | 0.467 / 0.0 | 0.800 / 0.0 |
| mismatch, mass +20% | adaptive / fallback | 0.400 / 0.0 | 0.467 / 0.0 |
| mismatch, severe | adaptive / fallback | 0.000 / 0.0 | 0.000 / 0.0 |
| ω kick, default ensemble | fallback | 0.333 / 0.0 | 0.400 / 0.0 |
| ω kick, narrow ensemble | fallback | 0.333 / **8.7** | 0.333 / **9.2** |

### Numbers: stratified success (n=15, K_min=100, K_max=300, σ=1.0, current controller)

| condition | fixed K=100 | fixed K=300 | adaptive (eff. K) |
|---|---|---|---|
| nominal | 86.7% | 60.0% | 93.3% (228.0) |
| mild mismatch | 46.7% | 13.3% | 53.3% (235.0) |
| severe mismatch | 0.0% | 0.0% | 6.7% (244.3) |
| state disturbance | 33.3% | 33.3% | 40.0% (230.3) |

### Numbers: OOD severity (15 repeats per level, ± z), equal to fig1

AUC by shift level: 0.0 → 0.520, 0.5 → 0.409, 1.0 → 0.458, 1.5 → 0.551,
2.0 → 0.654, 2.5 → 0.764, 3.0 → 0.895, 3.5 → 0.979, 4.0 → 0.999.

Uncertainty ratio, OOD over in-distribution: 1.03, 0.85, 0.94, 1.11, 1.38,
1.78, 2.39, 3.21, 4.27.

### Numbers: latency crossover (fresh process per K_max, 20 seeds, 30 steps, 1 repeat)

| K_max | fixed ms (published) | fixed ms (now) | adaptive ms (published) | adaptive ms (now) | ratio now |
|---|---|---|---|---|---|
| 100 | 28.9 | 31.8 | 665.1 | 808.1 | 0.039 |
| 1000 | 55.1 | 90.2 | 713.3 | 873.1 | 0.103 |
| 3000 | 151.1 | 259.0 | 736.1 | 684.7 | 0.378 |
| 8000 | 414.1 | 697.7 | 742.6 | 713.9 | 0.977 |
| 12000 | 577.2 | 1061.8 | 712.2 | 752.8 | 1.410 |
| 16000 | 838.3 | 1505.9 | 824.7 | 908.0 | 1.658 |
| 26000 | 1359.3 | 2448.3 | 821.5 | 790.7 | 3.096 |

- **Crossover location:** interpolated crossover K_max ≈ 8,210. The
  regression gives K* ≈ 8,687 (slope 0.0958 ms per rollout, R² = 0.998).
  The published values are 15,600 and 14,773.
- **Back-to-back check at K=16000, 2026-10-27:**
  - unmodified `legacy/adaptive_k_single_point.py`: fixed 1460.3 ms,
    adaptive 781.2 ms, ratio 1.869;
  - refactored `crossover_point`: fixed 1538.6 ms, adaptive 1159.7 ms,
    ratio 1.327.
- **GPU state at idle:** P8, 300 MHz SM clock, 62 °C. Under load: P3, 810 MHz,
  73 °C (maximum SM clock 1785 MHz).

### Numbers: ensemble MPPI inference benchmark (horizon 10, 20 commands)

| device | E | K | loop ms | batched ms | batched model evaluations per second |
|---|---|---|---|---|---|
| cuda | 5 | 1000 | 24.33 | 11.13 | 4.49 M |
| cuda | 5 | 500 | 23.78 | 11.48 | 2.18 M |
| cpu | 5 | 1000 | 44.88 | 52.07 | 0.96 M |

The paper's `tab:gpuspeedups` row 1 reports 23.65 → 12.37 ms and 4.04 M
evaluations per second.

### Observations

1. **[Strong] The refactor does not change any result.** The package
   reproduces every legacy script bit-for-bit, and every seeded published
   number at full scale.

2. **[Strong] Four published results were produced by the pre-fix
   controller:** `tab:confound`, `tab:mujoco` (regulation and reaching), and
   the narrow-ensemble fallback rate of 8.7%.
   - With `--pilot-rng shared` they reproduce exactly. With the current
     controller they do not.
   - On the mismatched seeds, even the single model's results differed,
     although it never uses the pilot. The old pilot consumed the shared
     global random stream during calibration, and that shifted the noise of
     every method evaluated after it.

3. **[Moderate, n=15] With the current controller, tab:mujoco's ordering
   changes.** Adaptive no longer beats ensemble at σ=1.0: it is 55.6% vs
   55.6%, where the paper has 62.2% vs 55.6%. On the reacher, adaptive drops
   from 33.3% to 30.0%. The paper's ensemble-over-single result survives,
   but its adaptive numbers in `tab:mujoco` are artifacts of the pre-fix
   random stream. Rerun the table with the current controller before
   resubmitting.

4. **[Moderate, n=20] With the current controller at σ=1.0, adaptive beats
   fixed-high at K=1000 and K=8000** (0.70 vs 0.40, 0.65 vs 0.45). This
   reverses the published `tab:confound` direction. It agrees with the
   paper's explanation that σ=1.0 handicaps large-K fixed controllers: the
   adaptive controller runs mostly at K=100, where σ=1.0 is near-optimal.

5. **[Weak to moderate, n=40, one ensemble] Adaptive escalates more under
   mismatch.** The rate is 6.56% under mild mismatch vs 2.25–2.88% at the
   nominal K_max values, with the same threshold of 0.008571. This seems to
   cut against the paper's statement that adaptive "does not escalate more
   under mismatch at all".
   - Caveat: the cases differ in K_max and ceiling σ, so the trajectories
     differ.
   - A same-K_max nominal-vs-mismatch comparison is needed before revising
     that sentence.
   - The nominal rates (2.3–2.9%) are also above the paper's quoted 1.7–1.8%.
     The paper's figure may count escalations differently; check how it was
     computed.

6. **[Strong] The 15,600 latency crossover is not stable on the machine that
   produced it.**
   - The same GPU and the unmodified legacy code now give fixed-high about
     1.8× slower at every K ≥ 1000. The compute-bound part slowed; the
     launch-overhead-bound adaptive part did not.
   - That moves the crossover to about 8,200–8,700.
   - Two back-to-back runs at K=16000 give ratios of 1.33 and 1.87.
   - The crossover location therefore depends on GPU power and thermal state
     as well as on hardware model. It needs repeated, clock-controlled
     measurements (median and IQR over ≥5 fresh processes) before any single
     number is claimed.

7. **[Moderate] The batched-inference speedup reproduces.** On CUDA at E=5,
   K=1000 it is 2.19×; the paper reports 1.91×. It does not help on CPU at
   K=1000 (44.9 → 52.1 ms), consistent with the paper.

8. **[Strong] Five paper numbers still have no producing script:**
   `tab:sigmasweep`, the 33.95× seed-batching row, two other rows of
   `tab:gpuspeedups`, and the Claim A K=1000 nominal row.

---

## 2026-10-30 · Derived statistics for `paper_arxiv/main.tex` (no new runs)

**What:** no experiment was rerun. All numbers below are recomputed from the
2026-10-26/27 records by `uv run python paper_arxiv/make_figures.py`, which
prints them. They use the current controller: `current_controller/` for the
four pilot-RNG-dependent experiments, `replicate/` for everything else.

### Numbers

- **Matched-ceiling paired sign tests (n=20):**
  - σ=1.0: K=1000 record 2 / 8 / 10, p=0.109; K=8000 record 3 / 7 / 10, p=0.344.
  - Retuned σ: K=1000 record 4 / 1 / 15, p=0.375; K=8000 record 3 / 3 / 14, p=1.000.
  - Records are fixed wins / adaptive wins / ties.
- **σ sweeps from the Claim A records (n=40, σ = 1–6):** these replace the
  unscripted `tab:sigmasweep`.
  - At the ceiling, σ=1.0 gives 17.5% / 12.5% / 12.5% at K = 3000 / 8000 / 16000.
  - At matched K = 403 and 682, every σ ≥ 2 reaches at least adaptive's 82.5%.
    At mismatch K=284, every σ ≥ 3 is above adaptive's 50%.
- **Adaptive escalation rate in the crossover sweep (σ=1.0):** 2.33–3.00% at
  every K_max from 300 to 26000.
- **Crossover adaptive latency:** 685–908 ms, CV 8.7%. Regression slope
  0.0958 ms/rollout, against 0.0525 in the paper (1.83×).
- **MuJoCo per-command latency (± t):**
  - regulation σ=1.0: single 42.1 ± 0.3 ms, ensemble 56.0 ± 0.3 ms, adaptive 57.5 ± 0.3 ms;
  - reacher: single 41.6 ms, ensemble 49.3 ms, adaptive 50.8 ms;
  - the pilot adds about 1.5 ms in each case.
- **Seed-level MuJoCo comparisons:**
  - Regulation σ=1.0: ensemble beats single on 7 / 15 seeds and loses on 3.
  - Reacher: worst-seed final distance is single 0.592, ensemble 0.275,
    adaptive 0.366.
- **OOD ensemble RMSE ratio (OOD / in-distribution):** 1.12, 0.95, 0.79, 0.71,
  0.70, 0.70, 0.75, 0.84, 0.93 for shift 0.0–4.0.
- **TPR at 5% FPR:** 0.041, 0.021, 0.064, 0.129, 0.285, 0.448, 0.622, 0.878,
  1.000.

### Observations

1. **[Moderate, one ensemble] Disagreement tracks input novelty, not
   prediction error.** The uncertainty ratio rises to 4.27× while the
   ensemble's RMSE ratio stays at or below 1.12×. The RMSE ratio is below 1 at
   every shift from 0.5 to 4.0.
2. **[Moderate] Low escalation is a property of calibration, not of K_max.**
   The escalation rate is about 2.5% at every K_max, below the 5% calibration
   target. The calibration states are uniform initial states, while
   closed-loop trajectories settle in the well-modelled upright region.
3. **[Moderate] The Claim A conclusion survives the best-σ selection on the
   test seeds.** Most σ values beat adaptive at matched K, so picking the best
   σ on the same seeds does not create the result.

## 2026-11-06 · Number check of `paper_arxiv/main.tex` (no new runs)

**What:** no experiment was rerun. Every number in `paper_arxiv/main.tex` was
compared with the output of `uv run python paper_arxiv/make_figures.py`, and
with the claim_a and crossover records for the σ-sweep table and the earlier
regression.

**Result:** all numbers agree, including:
- the Claim A, σ-sweep, matched-ceiling, fallback, MuJoCo, crossover and
  earlier-controller tables;
- the 28× and 41% figures, the 2.3–3.0% escalation range, and the 0.70–1.12
  RMSE ratios;
- the earlier crossover slope of 0.0525 ms/rollout, mean adaptive latency of
  767 ms, and K* = 14,773.

**Text corrections made:**
- distribution notation `NC(·)` → `\mathcal{U}(·)`;
- "accounts for every discrepancy" narrowed to seeded results, since the
  crossover shift has a separate cause (GPU state);
- the conclusion compared interpolation with regression (15,600 → 8,700). It
  now reads 15,600 → 8,200 by interpolation (14,773 → 8,687 by regression).

**Reproduction path corrected:**
- New suite `current_controller` produces the four isolated-controller
  results the paper quotes. The `replicate` suite produces their shared-stream
  versions instead.
- `make_figures.py` now takes `--runs`. With the default path it gives output
  identical to before.
- Appendix E updated to match.

**Not verified:** the paper has not been compiled, because no LaTeX
installation is available on this machine. Static checks pass: all
references, citations and figure files resolve, and environments and braces
balance.

### Observations

1. **[Strong] `paper_arxiv/main.tex` is consistent with the 2026-10-26/27 and
   2026-10-30 entries.** The older `.tex` files are not, and now carry a
   SUPERSEDED header comment.

## 2026-11-06 · First vast.ai smoke run: failed in setup (no results)

**Command:** `uv run ua-mbrl remote run --gpu RTX_4090 --suite smoke`. The
source was the uploaded working tree at `02802ec0c36d`.

**What happened:**
- **Machine:** offer 59203547, an RTX 4090 in Vietnam at $0.389/h, instance
  59377908. It was running within about 40 s and SSH worked.
- **Setup failed:** `set: pipefail\r: invalid option name`. Python on Windows
  sent the setup script over SSH in text mode, which converts line endings
  to CRLF.
- **Automatic destroy failed:** it stopped at vastai's interactive
  confirmation prompt. The instance was gone when checked afterwards.
  Cost: about $0.01.
- **Exit hung:** the run did not exit until Ctrl-C.

**Fixes in `src/ua_mbrl/remote/vast.py`, all covered by regression tests in
`tests/test_remote.py`:**
- the setup script is sent as raw LF bytes;
- `destroy instance -y`;
- no child process reads the console's stdin;
- the first Ctrl-C stops all workers cleanly: each fetches partial results
  and destroys its instance. A second Ctrl-C destroys every active instance
  immediately;
- `bootstrap.log` is saved even when setup fails;
- CN and VN offers are excluded by default (`--exclude-countries`), because
  hosts there were unreliable for the user in the past.

## 2026-11-06 · Second vast.ai smoke run: failed downloading CUDA wheels (no results)

**Command:** `uv run ua-mbrl remote run --gpu RTX_4090 --suite smoke`. The
source was the uploaded working tree at `788faaa5b548`.

**What happened:**
- **Machine:** offer 41644200, an RTX 4090 in Czechia at $0.394/h, instance
  59379186.
- **The previous fixes held:** setup ran, the failure was reported with
  `bootstrap.log` saved, the instance was destroyed automatically, and the
  run exited normally. Cost: about $0.04 over 0.10 h.
- **Setup failed:** `uv sync` could not download
  `nvidia-cufft-cu12==11.3.3.83` from `pypi.nvidia.com`. The error was
  "operation timed out", after uv's default 3 retries in 125 s.

**Fix:** the setup script now sets `UV_HTTP_TIMEOUT=600`,
`UV_HTTP_RETRIES=10` and `UV_CONCURRENT_DOWNLOADS=4`, and retries the whole
`uv sync` up to 4 times. Wheels already downloaded stay in uv's cache, so a
retry only fetches what is missing. Tested with a bash syntax check and a
dry run of the retry loop with a stand-in `uv`.

### Observations

1. **[Weak, one host] Download speed from a host to `pypi.nvidia.com` varies.**
   Setup time on vast.ai depends on the host's network as well as the GPU.
   Budget 10–20 min of setup per rental.

## 2026-11-06 · Third vast.ai smoke run: pipeline works end to end (RTX 4090)

**Command:** `uv run ua-mbrl remote run --gpu RTX_4090 --suite smoke`. The
source was the uploaded tree at commit `3f923f2`; `git_dirty` is false, so
provenance is exact from now on.

**Run directory:** `results/vast/rtx_4090/smoke-20261106-121402/`.

**Hardware:**
- GPU: RTX 4090 24 GB, driver 615.71.09, 3285 MHz maximum SM clock.
- CPU: AMD Ryzen 7 5700X, 8 cores.
- Host: France, offer 47697287, instance 59380360.
- Software: torch 2.11.0+cu128, the same build as the local GTX 1650.

**Result:**
- Rent, install, run, fetch and destroy all succeeded. The whole cycle took
  0.072 h, about $0.03.
- With the timeout and retry settings, `uv sync` finished on the first
  attempt on this host.
- All three smoke experiments exited with code 0, and the results were
  fetched.

**Numbers:** none of scientific value. The smoke configurations are tiny,
for example 3 seeds and 3 steps for Claim A, and 2 seeds and 2 steps for
the crossover. Do not compare them with full-scale results. In particular,
the crossover ratio of about 0.97 at K = 100 and 300 reflects S = 2 seeds,
not the paper's S = 20.

**Tooling follow-ups:**
- `remote instances` now says "no instances" when the list is empty.
- `ua-mbrl suite` compares only `replicate` and `replicate_quick` against
  the paper. `smoke` has no targets, and `current_controller` is meant to
  differ from them.

## 2026-11-06 · Full replication on an RTX 5090 (vast.ai)

**Command:** `uv run ua-mbrl remote run --gpu RTX_5090 --suite replicate`.
The source was the uploaded tree at commit `5ad4a60`, with `git_dirty`
false.

**Run directory:** `results/vast/rtx_5090/replicate-20261106-122442/`. The
statistics are printed by `uv run python paper_arxiv/make_figures.py` (its
`--second` option defaults to the latest RTX 5090 run).

**Hardware and software:**
- GPU: RTX 5090 32 GB, Blackwell, driver 610.43.02, 3090 MHz maximum SM
  clock.
- CPU: Intel Core Ultra 9 285K. Host in Mexico, offer 58979848.
- Software: torch 2.11.0+cu128, the same build as on the GTX 1650.

**Time and cost:** the whole suite took 0.274 h and cost about $0.17. The same
suite took about 2.6 h on the GTX 1650.

**Controller per experiment:** as in the `replicate` suite.
- Isolated (current controller): Claim A, matched ceiling, stratified, OOD,
  MuJoCo σ=0.2, MuJoCo matched budget, crossover.
- Shared (pre-fix): confound check, fallback, MuJoCo σ=1.0, reacher. Their
  isolated counterparts on the 5090 still need the `current_controller`
  suite (pending).

### Numbers: Claim A, matched sample budget (n=40, isolated)

| K_max | condition | adaptive (Wilson CI) | eff. K | escalation | fixed @ matched K (σ) | record F / A / T | p |
|---|---|---|---|---|---|---|---|
| 3000 | nominal | 75.0% [59.8, 85.8] | 306.6 | 2.81% | 87.5% @307 (σ 2) | 10 / 5 / 25 | 0.302 |
| 8000 | nominal | 80.0% [65.2, 89.5] | 466.9 | 3.06% | 95.0% @467 (σ 5) | 7 / 1 / 32 | 0.070 |
| 16000 | nominal | 82.5% [68.1, 91.3] | 652.3 | 2.69% | 95.0% @652 (σ 3) | 7 / 2 / 31 | 0.180 |
| 1000 | mild mismatch | 47.5% [32.9, 62.5] | 282.4 | 6.38% | 65.0% @282 (σ 4) | 11 / 4 / 25 | 0.118 |

- The ceiling controller reaches 100% at every nominal K_max and 75% under
  mismatch, as on the GTX 1650.
- The threshold is 0.008571, identical to the GTX 1650.
- Adaptive's per-seed outcomes agree with the GTX 1650 run on 88%, 92%, 90%
  and 82% of seeds.
- σ sweeps:
  - At the ceiling, σ=1 gives 30.0% / 22.5% / 15.0% at K = 3000 / 8000 / 16000.
  - At matched K: K=307 gives 55.0 / 87.5 / 87.5 / 82.5 / 87.5 / 80.0 for
    σ = 1–6; K=467 gives 60.0 / 82.5 / 82.5 / 87.5 / 95.0 / 87.5; K=652 gives
    45.0 / 85.0 / 95.0 / 90.0 / 90.0 / 92.5; mismatch K=282 gives
    32.5 / 45.0 / 52.5 / 65.0 / 57.5 / 65.0.

### Numbers: matched ceiling (n=20, isolated)

| K_max | σ=1.0, fixed / adaptive (F/A/T) | retuned σ, fixed / adaptive (F/A/T) |
|---|---|---|
| 100 | 0.75 / 0.75 (0/0/20) | 0.75 / 0.75 (0/0/20) |
| 1000 | 0.40 / 0.70 (2/8/10) | 0.80 / 0.70 (3/1/16) |
| 8000 | 0.45 / 0.70 (2/7/11) | 0.70 / 0.70 (3/3/14) |

The GTX 1650 gave 0.90 / 0.75 (4/1/15) at K=1000 retuned, and 0.45 / 0.65
(3/7/10) at K=8000 with σ=1.

### Numbers: latency crossover (fresh process per K_max, 20 seeds, 30 steps, 1 repeat)

| K_max | 100 | 1000 | 3000 | 8000 | 12000 | 16000 | 20000 | 26000 |
|---|---|---|---|---|---|---|---|---|
| fixed ms | 5.10 | 5.03 | 12.95 | 35.83 | 53.03 | 70.37 | 87.55 | 113.32 |
| adaptive ms | 95.35 | 96.90 | 94.93 | 96.63 | 100.28 | 96.01 | 95.66 | 102.92 |
| ratio | 0.053 | 0.052 | 0.136 | 0.371 | 0.529 | 0.733 | 0.915 | 1.101 |

- **Crossover:** interpolated K_max ≈ 22,737; regression K* ≈ 22,278.
- **Fit:** slope 0.00435 ms per rollout, intercept 0.64 ms, R² = 0.9999,
  n = 10.
- **Adaptive latency:** 94.4–102.9 ms, mean 97.5 ms, coefficient of
  variation 3.0%.
- **Escalation rate:** 2.3–3.0% at every K_max ≥ 300.
- **Fixed controller below K = 1000:** flat at 5.0–5.2 ms.

### Numbers: MuJoCo

| run | controller | single | ensemble | adaptive | latency ms (single / ens / adapt) |
|---|---|---|---|---|---|
| regulation σ=0.2, n=8 | isolated | 45.8 ± 29.6% | 62.5 ± 17.9% | 70.8 ± 31.4% (eff. K 195.8) | 5.47 / 7.85 / 8.17 |
| regulation σ=1.0, n=15 | shared | 37.8 ± 13.7% | 53.3 ± 16.8% | 66.7 ± 14.0% (eff. K 200.5) | 5.31 / 7.26 / 7.55 |
| reaching, n=10 | shared | 16.7 ± 11.2%, dist 0.347 | 35.0 ± 16.3%, dist 0.195 | 35.0 ± 16.3%, dist 0.207 | 5.36 / 6.71 / 6.99 |

MuJoCo matched budget, isolated: fixed 79.2 ± 25.5% vs adaptive
70.8 ± 31.4%, record 2 / 6 / 0, p = 0.50. Mean matched K is 195.9. The GTX
1650 gave 83.3% vs 70.8%, record 3 / 5 / 0.

### Numbers: other experiments

- **Bit-identical to the GTX 1650:**
  - the OOD sweep: every AUC and ratio;
  - the stratified study: every per-seed outcome, ESS trace and K choice;
  - `confound_check` at K=100 (0.75 / 0.35);
  - the fallback trigger rates, including 8.7% for the narrow ensemble.
- **`confound_check`, shared:** K=1000 gives 0.40 / 0.30 and K=8000 gives
  0.45 / 0.35. Published: 0.40 / 0.35 and 0.45 / 0.40.
- **Ensemble inference, E=5, horizon 10:**
  - CUDA, loop vs batched: 2.91 vs 1.59 ms at K=100, 2.93 vs 1.59 at K=500,
    2.93 vs 1.60 at K=1000. That is 1.83× at K=1000, with 31.2 M model
    evaluations per second batched.
  - CPU: 22.44 vs 20.62 ms at K=1000.

### Observations

1. **[Strong] Seeded results are bit-reproducible within a GPU, but not
   across GPU architectures, even with identical software.**
   - On identical code, torch, CUDA build and seeds, the GTX 1650 (Turing)
     and RTX 5090 (Blackwell) give identical calibration thresholds. The OOD
     sweep and the stratified study (K ≤ 300) also match bit-for-bit.
   - The closed-loop runs with large rollout batches differ. In Claim A, 8–18%
     of adaptive per-seed outcomes flip, and the MuJoCo per-seed records
     differ.
   - The most likely cause is float32 rounding differences between GPU
     kernels chosen for large batch shapes, amplified by the sensitive
     swing-up and regulation dynamics.
   - Consequence: "exact replication" holds only on the same GPU
     architecture. A run on another GPU behaves like an independent random
     draw of the same experiment.

2. **[Moderate] The Claim A direction replicates on the RTX 5090; its
   significance does not.**
   - In all four cases, fixed at the matched budget ≥ adaptive, and fixed has
     more paired wins (10/5, 7/1, 7/2, 11/4).
   - But no case reaches p < 0.05: the p-values are 0.302, 0.070, 0.180 and
     0.118. On the GTX 1650, K=16000 (p = 0.016) and mismatch (p = 0.031)
     did.
   - The per-case significance in the paper therefore depends on the exact
     float trajectory. Claims must rest on the consistent direction. More
     seeds or ensembles, as in plan Phase 1, are needed for per-case
     significance.

3. **[Moderate] "Adaptive scores exactly 82.5% at all three nominal budgets"
   was specific to the GTX 1650.** On the RTX 5090 it is 75.0%, 80.0% and
   82.5%. The mechanism holds: escalation stays at 2.7–3.1% of nominal steps
   and 6.4% under mismatch, and adaptive stays below the matched fixed
   controller.

4. **[Strong for these two machines, single repeat] The latency crossover
   depends on the hardware.**
   - It is K* ≈ 22,300–22,700 on the RTX 5090 host, against 8,200–8,700 on
     today's GTX 1650 and 15,600 in the earlier GTX 1650 measurement.
   - The fixed controller's per-rollout slope is 22× smaller
     (0.00435 vs 0.0958 ms per rollout).
   - The adaptive controller's flat latency is only 8.2× smaller
     (97.5 vs 795.6 ms). It is bound by launch and CPU overhead, so it
     improves less, and the crossover moves up by 2.6×.
   - Below K ≈ 1000 the RTX 5090's fixed controller is entirely launch-bound
     (flat 5 ms).
   - Adaptive latency is much steadier on this host (coefficient of
     variation 3.0% vs 8.7%).

5. **[Moderate] Batched inference still helps on a current GPU.** It is 1.83×
   faster on CUDA at K = 1000, against 2.19× on the GTX 1650. Throughput is
   7× higher (31.2 M vs 4.49 M evaluations per second). CUDA command latency
   is flat from K=100 to K=1000 (about 1.6 ms), so at these sizes the
   controller is launch-bound on the RTX 5090.

6. **[Weak, n=8] MuJoCo σ=0.2 ensemble vs single shifts.** On the RTX 5090,
   adaptive (70.8%) is above the ensemble (62.5%); on the GTX 1650 both were
   70.8%. The matched-budget direction holds (fixed ≥ adaptive on every
   seed). With intervals of about ±30 points, none of these differences is
   informative.

## 2026-11-06 · Current-controller suite on an RTX 5090 (vast.ai)

**Command:** `uv run ua-mbrl remote run --gpu RTX_5090 --suite current_controller`.
The source was the uploaded tree at commit `9c15260`, with `git_dirty`
false.

**Run directory:** `results/vast/rtx_5090/current_controller-20261106-132115/`.
The statistics are printed by `paper_arxiv/make_figures.py` (its
`--second-current` option defaults to the latest such run).

**Hardware:**
- GPU: RTX 5090 32 GB, driver 580.178.04.
- CPU: AMD Ryzen 7 7800X3D, 8 cores.
- Host: South Korea, offer 45669277, instance 59386519.
- This is a different host from the `replicate` run, which used an Intel
  Core Ultra 9 285K with driver 610.43.02.

**Time and cost:** 0.17 h, about $0.09. Most of the time was the four
experiments themselves (about 7 min).

All four experiments use the current controller (isolated pilot). They are
compared with the GTX 1650's `current_controller/` run.

### Numbers

- **Confound check (σ=1, n=20):**

  | K_max | fixed / adaptive | record F / A / T |
  |---|---|---|
  | 100 | 0.75 / 0.75 | 0 / 0 / 20 |
  | 1000 | 0.40 / 0.70 | 2 / 8 / 10 |
  | 8000 | 0.45 / 0.70 | 2 / 7 / 11 |

  The GTX 1650 gave 0.45 / 0.65 at K=8000; the rest is the same.
- **Fallback:** identical to the GTX 1650 current-controller run. Mismatch:
  success 0.800 / 0.467 / 0.000, fired 0%. Default ω kick: 0.400, fired 0%.
  Narrow ensemble: 0.333, fired 9.2%.
- **MuJoCo regulation σ=1.0 (n=15, ± t):**

  | method | success | latency | same success as GTX 1650 |
  |---|---|---|---|
  | single | 40.0 ± 14.3% | 7.39 ms | 10/15 seeds |
  | ensemble | 60.0 ± 14.3% | 10.95 ms | 7/15 seeds |
  | adaptive | 55.6 ± 16.6% (eff. K 203.9) | 11.32 ms | 13/15 seeds |

  The ensemble beats the single model on 10 seeds and loses on 3.
- **Reaching (n=10, ± t):**

  | method | success | final distance | worst seed | latency |
  |---|---|---|---|---|
  | single | 16.7 ± 11.2% | 0.351 ± 0.098 | 0.601 | 7.12 ms |
  | ensemble | 35.0 ± 16.3% | 0.194 ± 0.034 | 0.289 | 9.21 ms |
  | adaptive | 30.0 ± 11.0% | 0.214 ± 0.045 | 0.377 | 9.60 ms |

  Success on every seed matches the GTX 1650 for all three methods; only the
  distances differ slightly.

### Observations

1. **[Moderate, two GPUs] With the current controller, adaptive does not
   beat the ensemble on MuJoCo regulation.** It is 55.6% vs 60.0% here, and
   55.6% vs 55.6% on the GTX 1650. This confirms on independent hardware
   that `tab:mujoco`'s 62.2% (adaptive ahead) came from the pre-fix
   controller.
   - The ensemble's advantage over the single model holds on both GPUs:
     here 60.0% vs 40.0% (10 seeds better, 3 worse); on the GTX 1650 55.6%
     vs 44.4% (7 better, 3 worse).

2. **[Moderate] The reaching results are the most hardware-robust MuJoCo
   result.** Success matches on every seed and method across GPUs. The
   ensemble's worst-seed distance (0.289 vs the single model's 0.601) again
   about halves the single model's.

3. **[Moderate, one comparison] Per-command latency depends on the host, not
   only the GPU model.**
   - On the same GPU model, this host (Ryzen 7 7800X3D, driver 580) is about
     1.4× slower per MuJoCo command than the `replicate` host (Core Ultra 9
     285K, driver 610). For regulation at σ=1 the ensemble is 10.95 vs
     7.26 ms; single 7.39 vs 5.31; adaptive 11.32 vs 7.55.
   - That run used the pre-fix pilot, which does not change the amount of
     compute.
   - At these sizes the controller is bound by launch and CPU overhead, so
     the host CPU and driver set the latency.
   - Consequence for the latency study: compare GPUs only within one rental,
     with repeats, and record the host CPU. The crossover on the "same" GPU
     model can move with the host.

4. **[Strong] The fallback and confound conclusions transfer across GPUs.**
   The fallback never fires under parameter mismatch, and fires on 9.2% of
   steps for the narrow ensemble after a state kick. At σ=1, adaptive's
   apparent lead over fixed-high at K=1000 and 8000 appears on both GPUs.

## 2026-11-06 · Latency study: repeated crossover sweeps on RTX 4090 and RTX 5090 (vast.ai)

**Command:**
`uv run ua-mbrl remote run --gpu RTX_4090 RTX_5090 H100_SXM --exclusive --experiment crossover_sweep -- --repeats 5 --k-values 100 300 1000 3000 8000 16000 24000 32000 48000 64000`.
The source was the uploaded tree at commit `bbca69e`.

**Run directories:**
- `results/vast/rtx_4090/crossover_sweep-20261106-140033/`
- `results/vast/rtx_5090/crossover_sweep-20261106-140033/`

The statistics are printed by `paper_arxiv/make_figures.py` (section
"Latency study").

**Machines:** whole-machine rentals (`--exclusive`), run in parallel.

| GPU | driver | host CPU | location | time | cost |
|---|---|---|---|---|---|
| RTX 4090 | 595.99.02 | AMD Ryzen 5 7600 | Brazil | 0.19 h | $0.09 |
| RTX 5090 | 580.178.04 | AMD Ryzen 7 7800X3D | South Korea | 0.18 h | $0.10 |

**H100 SXM:** skipped, with no rental and no cost. There was no
whole-machine H100 SXM offer at the time.

**Protocol:** one fresh process per (K_max, repeat), 5 repeats, 20 seeds,
30 control steps, current controller. The regression uses K_max ≥ 1000
(8 points).

### Numbers

All five crossover measurements so far, in ms:

| machine | repeats | fixed slope a (ms/rollout) | intercept b | adaptive mean | K* regression | K* interpolation | per-repeat K* range |
|---|---|---|---|---|---|---|---|
| GTX 1650 laptop | 1 | 0.09584 | −36.98 | 795.6 | 8,687 | 8,210 | – |
| RTX 5090, host A (Core Ultra 9 285K, driver 610) | 1 | 0.00435 | 0.64 | 97.5 | 22,278 | 22,737 | – |
| RTX 5090, host B (Ryzen 7 7800X3D, driver 580) | 5 | 0.00433 | 1.73 | 135.8 | 30,994 | 30,807 | 30,880–31,183 |
| RTX 4090 (Ryzen 5 7600, driver 596) | 5 | 0.00717 | 0.96 | 130.7 | 18,090 | 17,593 | 17,873–18,408 |

Host A is the 5090 from the 2026-11-06 replicate run, a single repeat over
K_max = 100 to 26,000.

**Repeatability within one machine:**
- The largest coefficient of variation across 5 repeats is 1.68% for fixed
  and 4.15% for adaptive on the RTX 4090, and 0.86% and 2.70% on the RTX
  5090.
- The fixed controller's IQR is ≤ 0.05 ms at every K ≥ 300.

**Median latency in ms, fixed / adaptive, by K_max:**

| K_max | RTX 4090 | RTX 5090 (host B) |
|---|---|---|
| 100 | 6.16 / 127.34 | 6.60 / 134.74 |
| 1,000 | 6.34 / 127.27 | 6.63 / 135.04 |
| 3,000 | 23.17 / 127.79 | 13.73 / 134.45 |
| 8,000 | 59.19 / 127.16 | 36.63 / 134.37 |
| 16,000 | 116.23 / 127.62 | 71.24 / 134.47 |
| 24,000 | 173.28 / 128.30 | 105.54 / 134.76 |
| 32,000 | 230.27 / 130.20 | 139.94 / 134.60 |
| 48,000 | 344.64 / 134.87 | 209.24 / 136.39 |
| 64,000 | 460.02 / 140.12 | 278.66 / 139.86 |

### Observations

1. **[Strong for this decomposition, two hosts] The crossover splits into a
   GPU term and a host term.**
   - The fixed controller's per-rollout slope is a property of the GPU. On
     the same RTX 5090 model it was 0.00435 and 0.00433 ms per rollout on two
     different hosts, 0.5% apart.
   - The adaptive controller's flat latency is a property of the host CPU
     and driver: 97.5 vs 135.8 ms on those two hosts (1.39×). That matches
     the 1.4× MuJoCo per-command gap between the same two hosts.
   - Because K* ≈ (T_adapt − b) / a, the same GPU model gives a crossover of
     22,300 on one host and 31,000 on the other.
   - Across GPUs, the RTX 4090 has a 1.65× steeper slope than the RTX 5090
     (0.00717 vs 0.00433), so with a similar host its crossover is lower
     (18,100).

2. **[Strong] Within one machine and session, the crossover is repeatable.**
   Per-repeat estimates stay within −1% to +2% of the median on the RTX 4090
   and within 1% on the RTX 5090.
   - The 41% back-to-back variation and the 2× drift seen on the GTX 1650 are
     therefore properties of that laptop's power and thermal state, not of
     the method.
   - On a dedicated machine a single number is well defined, but it is a
     number for that GPU and host, not for "the method".

3. **[Moderate] Adaptive latency grows slightly at very large K_max.** It
   rises from 127 to 140 ms on the RTX 4090 and from 135 to 140 ms on the RTX
   5090 between K_max = 16,000 and 64,000. The 2–3% of escalated steps start
   to cost real compute. The affine-vs-flat model is therefore approximate
   above the crossover; it is slightly optimistic for adaptive.

4. **[Moderate] Below about 1,000 rollouts every modern-GPU fixed controller
   is launch-bound.** Latency is flat at 6.2–6.6 ms, or 5.0–5.2 ms on host A.
   There, the fixed controller is about 20× faster than adaptive (ratio
   0.048–0.053 on all three modern measurements).

5. **[Tooling] No exclusive H100 SXM offer was available.** Rerun the H100
   without `--exclusive`, or use `A100_SXM4`, for a datacenter data point.
