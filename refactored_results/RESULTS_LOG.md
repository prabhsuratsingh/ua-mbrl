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
