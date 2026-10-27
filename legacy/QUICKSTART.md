# Quick Start Guide

## Objective

Implement Phase 2 of UA-MPPI research: **Ensemble dynamics with uncertainty quantification**.

This validates that:
1. ✓ Ensemble disagreement correlates with model error (research question)
2. ✓ Ensemble MPPI can control pendulum (technical validation)
3. ✓ Foundation ready for uncertainty-aware cost (Phase 3)

## What Was Added

### New Files
- `ensemble_dynamics.py` — Ensemble model implementation
- `ensemble_mppi.py` — Ensemble MPPI controllers
- `experiment_ensemble_uncertainty.py` — Research experiment 1
- `validation_ensemble_mppi.py` — Research experiment 2
- `IMPLEMENTATION_GUIDE.md` — Detailed documentation
- `QUICKSTART.md` — This file

### Modified
- Nothing! Fully backward compatible with existing code

## Run the Experiments

### Setup
```bash
cd d:\Reinforcement-Learning\ua-mbrl
# Ensure torch, numpy, matplotlib, scipy are installed
```

### Experiment 1: Does Ensemble Uncertainty Predict Error?

```bash
python experiment_ensemble_uncertainty.py
```

**What it does:**
- Trains single DNN model
- Trains ensemble of 5 models
- Measures: ensemble disagreement vs. actual prediction error
- Computes correlation statistics
- Generates plots

**Expected output:**
```
UNCERTAINTY vs ERROR CORRELATION
Pearson r:  0.75 (p=1.23e-45)
✓ SUCCESS: Ensemble can predict model failure regions
```

**Interpretation:**
- If r > 0.7: ✓ Ensemble is good uncertainty estimator
- If 0.5 < r < 0.7: ~ Moderate predictive value
- If r < 0.5: ✗ Need alternative approach

**Output:** `runs/uncertainty_vs_error.png`, etc.

---

### Experiment 2: Does Ensemble MPPI Work on Real Control?

```bash
python validation_ensemble_mppi.py
```

**What it does:**
- Trains single DNN and ensemble
- Runs MPPI control with single model (10 episodes)
- Runs MPPI control with ensemble (10 episodes)
- Compares success rates and trajectories

**Expected output:**
```
SINGLE MODEL MPPI
Success rate: 90.0% (9/10)
Mean |theta error|: 0.045

ENSEMBLE MPPI (no uncertainty)
Success rate: 90.0% (9/10)
Mean |theta error|: 0.045

Key Comparison:
  Single model success rate: 90.0%
  Ensemble success rate:     90.0%
  Ensemble average uncertainty: 0.000234
```

**Output:** `runs/ensemble_mppi_comparison.png`, etc.

---

## What to Look For

### Key Results

| Metric | Expected | Pass? |
|--------|----------|-------|
| Correlation (r) | > 0.7 | ✓ if yes |
| Single success | > 80% | ✓ if yes |
| Ensemble success | ≈ single | ✓ if similar |
| Uncertainty > 0 | Yes | ✓ if nonzero |

### Good Signs
- ✓ Ensemble and single model have similar success rates
- ✓ Ensemble uncertainty is nonzero (~0.0001)
- ✓ Correlation between uncertainty and error is strong
- ✓ No convergence issues during training

### Bad Signs
- ✗ Ensemble has much lower success rate than single
- ✗ Ensemble uncertainty is always zero
- ✗ Correlation is very weak (r < 0.3)
- ✗ Training diverges or gets NaN

---

## Understanding the Architecture

```
Data Collection
     ↓
┌────────────────────┐
│ Single DNN Model   │  vs  │ Ensemble (5 models) │
│ Input: [sin(θ), cos(θ), ω, u]                │
│ Output: [Δθ, Δω]                             │
└────────────────────┘
     ↓
┌───────────────────────────────────────────┐
│ One-Step Prediction Error Evaluation      │
│ Compare: ensemble variance vs error       │
└───────────────────────────────────────────┘
     ↓
     Correlation Analysis
     
     r = ?
     ├─ r > 0.7: ✓ Good uncertainty estimator
     └─ r < 0.5: ✗ Need alternative approach
```

---

## Next Phase (When Ready)

Once experiments validate ensemble uncertainty:

### Phase 3: Uncertainty-Aware MPPI Cost

Modify the cost function:
```python
# Standard MPPI cost
J_task = θ_error² + ω_error² + u_weight*u²

# Add uncertainty penalty
J_uncertainty = β * trace(ensemble_covariance)

# Combined
J_total = J_task + β * J_uncertainty
```

This makes MPPI prefer trajectories with higher model confidence.

### Phase 4-5: Adaptive Computation

Adjust number of samples K based on uncertainty:
```
High uncertainty → Use more samples (K=5000)
Low uncertainty  → Use fewer samples (K=500)
```

This balances performance with computational cost.

---

## File Locations

```
d:\Reinforcement-Learning\ua-mbrl\

Core Files (Original)
├── mppi_pendulum_learning_gpu.py    (GPU single model + MPPI)
├── mppi_pendulum_learning.py        (CPU version)

New Implementation (Phase 2)
├── ensemble_dynamics.py              (Ensemble models)
├── ensemble_mppi.py                  (Ensemble controllers)
├── experiment_ensemble_uncertainty.py (Research validation 1)
├── validation_ensemble_mppi.py        (Research validation 2)

Documentation
├── IMPLEMENTATION_GUIDE.md           (Detailed guide)
├── QUICKSTART.md                     (This file)
├── README.md                         (Project overview)

Outputs
└── runs/                             (Generated plots)
    ├── uncertainty_vs_error.png
    ├── ensemble_mppi_comparison.png
    └── ... (other plots)
```

---

## Troubleshooting

### Experiment takes too long
- Reduce `TRAIN_EPOCHS`: 100 → 50
- Reduce `MPPI_ROLLOUTS`: 1000 → 500
- Reduce `EVAL_EPISODES`: 10 → 3

### "CUDA out of memory"
- Reduce `ENSEMBLE_SIZE`: 5 → 3
- Reduce rollouts (K): 1000 → 500
- Add `torch.cuda.empty_cache()` before MPPI

### Low correlation (r < 0.5)
- Check ensemble diversity: print predictions from each model
- Verify test data covers full range
- Try larger ensemble (E=10)
- Check network architecture (might be underfitting)

### Import errors
- Ensure all new files are in `d:\Reinforcement-Learning\ua-mbrl\`
- Check Python path: `import sys; sys.path.insert(0, os.path.dirname(__file__))`
- Verify torch/numpy installed

---

## Key Insights from the Proposal

From the attached research proposal:

1. **Incremental approach**: Single → Ensemble → Uncertainty → Adaptive K
2. **Algorithm first**: Validate each phase before GPU optimization
3. **Clear experiments**: Each phase has specific research question
4. **Pendulum as dev environment**: Quick iteration before MuJoCo/quadrotor
5. **System contribution**: Adaptive computation (Phase 5) is novel

This implementation focuses on Phase 2: building the ensemble and validating uncertainty quantification.

---

## Expected Paper Contribution

Once all phases complete, paper would be:

**Title**: "Uncertainty-Adaptive GPU-Parallel Model Predictive Path Integral Control with Online Learned Dynamics"

**Contributions**:
1. Learned dynamics model for MPPI (vs. analytical)
2. Ensemble uncertainty quantification
3. Uncertainty-aware MPPI objective
4. Adaptive trajectory sampling based on uncertainty
5. GPU implementation enabling real-time control
6. Comprehensive evaluation across environments

This implementation focuses on **Contributions 1-2**, with infrastructure for **3-4**.

---

## Questions?

- Check `IMPLEMENTATION_GUIDE.md` for detailed architecture
- Review individual experiment files for specific algorithms
- Refer to original proposal document for research context

Good luck! 🚀
