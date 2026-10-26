# Uncertainty-Adaptive GPU Model-Based MPPI: Ensemble Implementation

This document describes the **Phase 1-3 implementation** of the UA-MPPI research framework focused on ensemble dynamics and uncertainty quantification.

## Overview

This codebase implements a research progression for uncertainty-aware model-based reinforcement learning with MPPI control:

```
Phase 1 (✓ Current)   → Single DNN + GPU MPPI
         ↓
Phase 2 (✓ Current)   → Ensemble dynamics + uncertainty estimation
         ↓
Phase 3 (Ready)       → Uncertainty-aware MPPI cost
         ↓
Phase 4 (Planned)     → Online adaptive ensemble
         ↓
Phase 5 (Planned)     → Adaptive trajectory sampling (K=f(uncertainty))
```

## New Modules

### 1. `ensemble_dynamics.py`

Implements probabilistic ensemble of neural network dynamics models.

**Key Classes:**

- **`DynamicsModel`**: Single deterministic dynamics model
  - Input: `[sin(θ), cos(θ), ω, u]` → Output: `[Δθ, Δω]`
  - Methods: `forward()`, `predict()`, `model_step()`, `train_model()`
  
- **`EnsembleDynamicsModel`**: Ensemble of E independent models
  - Trains E separate models on the same dataset
  - Computes mean and variance predictions
  - Methods:
    - `forward(x, u)` → `[E, B, 2]` (all predictions)
    - `forward_mean_var(x, u)` → `(mean, var, std)` per sample
    - `predict_uncertainty(x, u)` → scalar uncertainty per sample
    - `model_step_with_uncertainty(x, u)` → mean + uncertainty
    - `evaluate_ensemble_disagreement(test_data)` → diagnostic dict

**Research Value:**
- Epistemic uncertainty via model disagreement
- Uncertainty correlated with prediction error
- Foundation for uncertainty-aware control

### 2. `ensemble_mppi.py`

GPU-vectorized MPPI controllers with ensemble support.

**Key Classes:**

- **`EnsembleMPPIController`**: Main controller
  - Accepts ensemble or single model
  - Modes:
    - Standard: `use_uncertainty=False` (ensemble mean only)
    - Uncertainty-aware: `use_uncertainty=True` (cost includes uncertainty)
  - Cost function: `J_k = J_k^task + β * J_k^uncertainty`
  - Ready for adaptive K in Phase 5
  
- **`VanillaMPPIController`**: Baseline MPPI
  - Standard MPPI without uncertainty
  - Used for comparison

**Key Methods:**
- `command(x0, diagnose=False)` → control action
- Returns diagnostic info on request (imagined errors, uncertainties)

### 3. `experiment_ensemble_uncertainty.py`

**Research Experiment**: "Does ensemble disagreement predict model failure?"

**Purpose:**
Answer the core research question: Can ensemble variance quantify uncertainty well enough to indicate where the model will fail?

**Workflow:**
1. Collect random pendulum data
2. Train single DNN model
3. Train ensemble (E=5 models)
4. Evaluate one-step predictions on test set
5. Compute ensemble disagreement (variance)
6. Correlate uncertainty with prediction error
7. Visualize relationships

**Key Outputs:**
- `runs/uncertainty_vs_error.png`: Scatter plot with correlation
- `runs/uncertainty_distribution.png`: Histograms
- `runs/ensemble_std_vs_error.png`: Per-dimension analysis
- Console: Pearson & Spearman correlation statistics

**Interpretation:**
- **Strong correlation (r > 0.7)**: ✓ Ensemble is good uncertainty estimator
  - → Proceed to Phase 3 (uncertainty-aware MPPI)
- **Moderate correlation (0.5 < r < 0.7)**: ~ Some predictive value
  - → Consider alternative uncertainty estimators
- **Weak correlation (r < 0.5)**: ✗ Ensemble disagreement alone insufficient
  - → Need epistemic + aleatoric uncertainty separation

### 4. `validation_ensemble_mppi.py`

**Research Experiment**: "Does ensemble MPPI work on real control?"

**Purpose:**
Validate that ensemble-based MPPI can control the pendulum successfully and that uncertainty estimates are available during control.

**Workflow:**
1. Collect random pendulum data
2. Train single DNN model  
3. Train ensemble (E=5 models)
4. Run MPPI control with single model (10 episodes)
5. Run MPPI control with ensemble (10 episodes)
6. Compare success rates, control effort, trajectory quality
7. Visualize: side-by-side comparisons

**Key Outputs:**
- `runs/ensemble_mppi_comparison.png`: Trajectory comparison
- `runs/ensemble_summary_stats.png`: Bar chart metrics
- Console: Success rates, mean errors, control statistics

**Metrics Tracked:**
- Success rate (θ error < 0.1 rad AND ω < 0.5 rad/s)
- Final θ error
- Mean |u| (control effort)
- Model error magnitude during rollout
- Predicted uncertainty magnitude

## How to Run the Experiments

### Prerequisites
```bash
cd d:\Reinforcement-Learning\ua-mbrl
# Environment should have: torch, numpy, matplotlib, scipy
```

### Experiment 1: Ensemble Uncertainty Analysis

**Command:**
```bash
python experiment_ensemble_uncertainty.py
```

**Expected Runtime:** ~2-5 minutes (depends on GPU)

**Expected Output:**
```
Compute device: cuda
============================================================
STEP 1: COLLECTING INITIAL DATA
============================================================
Total samples: 2000
Train samples: 1600
Test samples: 400
...
============================================================
STEP 3: TRAINING ENSEMBLE (5 models)
============================================================
--- Training model 1/5 ---
Epoch:    0 | Loss: ...
Epoch:   10 | Loss: ...
...
============================================================
UNCERTAINTY vs ERROR CORRELATION
============================================================
Pearson r:  0.XXXX (p=Y.XXe-XX)
Spearman r: 0.XXXX (p=Y.XXe-XX)

Key Research Finding:
  Correlation between ensemble uncertainty and error: r=0.XXXX
  ✓ SUCCESS: Ensemble can predict model failure regions
  → Proceed to uncertainty-aware MPPI
```

**Output Files:**
- `runs/uncertainty_vs_error.png`
- `runs/uncertainty_distribution.png`
- `runs/ensemble_std_vs_error.png`

### Experiment 2: Ensemble MPPI Validation

**Command:**
```bash
python validation_ensemble_mppi.py
```

**Expected Runtime:** ~5-10 minutes (depends on GPU and K=1000)

**Expected Output:**
```
Compute device: cuda
============================================================
STEP 5: SINGLE-MODEL MPPI
============================================================
Episode  1/10: theta_err=  0.050 | omega=  0.100 | success=True
...
============================================================
SINGLE MODEL MPPI
============================================================
Success rate: 90.0% (9/10)
Mean |theta error|: 0.0450
Mean |omega|: 0.1200
...

============================================================
STEP 6: ENSEMBLE MPPI
============================================================
...

Key Comparison:
  Single model success rate: 90.0%
  Ensemble success rate:     90.0%
  Ensemble average uncertainty: 0.000234
```

**Output Files:**
- `runs/ensemble_mppi_comparison.png`
- `runs/ensemble_summary_stats.png`

## Expected Results

### Experiment 1: Uncertainty vs Error Correlation

**Hypothesis:** Ensemble disagreement correlates with model prediction error.

**Expected Results:**
- Pearson r: 0.65 - 0.85 (depending on model capacity and data quality)
- p-value: < 0.001 (highly significant)
- Scatter plot shows positive trend
- Higher uncertainty → Higher prediction error

**Interpretation:**
- Ensemble variance captures epistemic (model) uncertainty
- Can guide uncertainty-aware control decisions
- Ready for Phase 3

### Experiment 2: Ensemble MPPI Validation

**Hypothesis:** Ensemble mean predictions perform similarly to single model.

**Expected Results:**
- Single model success rate: ~85-95%
- Ensemble success rate: ~85-95% (similar or slightly better)
- Mean θ error: < 0.1 rad
- Mean control effort: similar between single and ensemble
- Ensemble uncertainty magnitude: ~0.0001 - 0.001

**Interpretation:**
- Ensemble doesn't degrade control performance
- Uncertainty estimates available for Phase 3
- Foundation solid for uncertainty-aware cost term

## Architecture Diagram

```
Current Implementation:

┌─────────────────────────────────────────┐
│      Pendulum Environment (True)        │
│  θ̈ = -(g/l)sin(θ) + u/(ml²)            │
└──────────┬──────────────────────────────┘
           │ (true transitions)
           ↓
      [Data Buffer]
           │
    ┌──────┴──────┐
    ↓             ↓
┌──────────┐  ┌──────────────┐
│ Single   │  │ Ensemble     │
│ DNN      │  │ of 5 DNNs    │
└──────────┘  └──────────────┘
    │              │
    │          ┌───┴────┐
    │          ↓        ↓
    │    [mean] + [variance]
    │          
    ↓          ↓
┌─────────────────────────────┐
│  EnsembleMPPIController     │
│  - Vectorized horizon loop  │
│  - K batched trajectories   │
│  - GPU tensors              │
└──────────────┬──────────────┘
               │ u_t
               ↓
          [True Environment]
               │
      ┌────────┴────────┐
      ↓                 ↓
  [Error]           [Success/Failure]
```

## Phase 3: Uncertainty-Aware MPPI (Ready to Implement)

Once correlation is validated, Phase 3 adds uncertainty to the MPPI cost:

```python
# Phase 3 modification (in EnsembleMPPIController.command)
task_cost = (theta_error² + omega_error² + u_weight*u²) * dt
uncertainty_cost = β * uncertainty * dt  # β is tunable weight

total_cost = task_cost + uncertainty_cost
```

This implements:
$$J_k = J_k^{task} + \beta J_k^{unc}$$

where:
- $J_k^{task}$ = standard MPPI objective
- $J_k^{unc} = \sum_i \text{tr}(\Sigma_{k,i})$ = trace of covariance
- $\beta$ = uncertainty weight (tunable for exploration/exploitation)

## Phase 5: Adaptive K (Future)

Once uncertainty is in cost, can adapt number of samples:

```python
uncertainty_mean = np.mean(ensemble_uncertainties_over_horizon)
K_t = K_min + (K_max - K_min) * sigmoid(α * (uncertainty_mean - τ))
```

This gives:
$$\text{uncertainty} \rightarrow \text{compute allocation}$$

## File Structure

```
ua-mbrl/
├── mppi_pendulum_learning_gpu.py       (original single model + MPPI)
├── mppi_pendulum_learning.py           (CPU version)
├── ensemble_dynamics.py                (NEW: ensemble models)
├── ensemble_mppi.py                    (NEW: ensemble controllers)
├── experiment_ensemble_uncertainty.py  (NEW: research validation 1)
├── validation_ensemble_mppi.py         (NEW: research validation 2)
├── pyproject.toml
├── README.md
└── runs/                               (outputs)
    ├── uncertainty_vs_error.png
    ├── uncertainty_distribution.png
    ├── ensemble_std_vs_error.png
    ├── ensemble_mppi_comparison.png
    └── ensemble_summary_stats.png
```

## Next Steps (Recommended Order)

1. **Run Experiment 1** (`experiment_ensemble_uncertainty.py`)
   - Validate ensemble uncertainty hypothesis
   - If r > 0.7: ✓ proceed to Phase 3
   - If r < 0.5: investigate uncertainty sources

2. **Run Experiment 2** (`validation_ensemble_mppi.py`)
   - Verify ensemble MPPI works on real control task
   - Check success rates match single model
   - Confirm uncertainty magnitude reasonable

3. **Implement Phase 3**
   - Add uncertainty term to cost: `J_k = J_k^task + β * J_k^unc`
   - Experiment with different β values (0.1, 1.0, 10.0, ...)
   - Measure if uncertainty helps with robustness

4. **Test Robustness** (Model Mismatch)
   - Train on m=1.0, test on m=1.2 (mass change)
   - Train on g=9.81, test on g=8.5 (gravity change)
   - Compare: vanilla MPPI vs uncertainty-aware MPPI
   - Expected: uncertainty-aware should fail more gracefully

5. **Implement Phase 4-5**
   - Online adaptive ensemble
   - Adaptive K based on uncertainty
   - Measure computational savings

## Key Research Questions

### Phase 2 (Current)
1. **Does ensemble disagreement predict error?**
   - Metric: Pearson correlation (target: > 0.7)
   - File: `experiment_ensemble_uncertainty.py`

2. **Does ensemble MPPI control well?**
   - Metric: Success rate, mean error
   - File: `validation_ensemble_mppi.py`

### Phase 3 (Next)
3. **Does uncertainty help with robustness?**
   - Compare vanilla vs uncertainty-aware MPPI
   - Under model mismatch, disturbances

### Phase 4-5 (Future)
4. **Can we reduce compute while maintaining performance?**
   - Measure: Sample efficiency, wall-clock time
   - Under uncertainty-guided adaptive K

## Debugging & Troubleshooting

### Issue: Low correlation in Experiment 1
**Possible Causes:**
- Ensemble models converged to same solution (try different random seeds)
- Test set not representative (check data ranges)
- Network capacity too low (increase hidden layer size)
- Not enough ensemble diversity (E=5 might be insufficient)

**Solutions:**
- Visualize predictions from each ensemble member
- Check weight diversity across ensemble
- Try E=10, E=20 for stronger disagreement
- Consider separate data for training each model

### Issue: Ensemble MPPI fails when single model succeeds
**Possible Causes:**
- Averaging predictions damages performance (shouldn't happen)
- Uncertainty estimate is zero (ensemble too similar)
- Bug in forward_mean_var()

**Solutions:**
- Verify ensemble.forward() returns [E, B, 2] correctly
- Print ensemble predictions for same input: should differ
- Check uncertainties > 0: `np.std(predictions, axis=0)`

### Issue: GPU out of memory
**Solutions:**
- Reduce K (rollouts): 1000 → 500
- Reduce N (horizon): 30 → 15
- Reduce batch size in training
- Reduce ensemble size: 5 → 3

## References to Original Proposal

This implementation follows the Phase 1-2 roadmap from the research proposal:

- **Phase 1**: ✓ Existing single model MPPI
- **Phase 2**: ✓ Ensemble dynamics + uncertainty estimation (current)
- **Phase 3**: Ready, needs: uncertainty-aware cost term
- **Phase 4**: Planned, needs: online ensemble retraining
- **Phase 5**: Planned, needs: adaptive K sampling

The proposal emphasizes building incrementally with clear research questions at each phase, which this implementation follows.

## Contact & Notes

This is a research framework for exploring uncertainty-aware model-based RL. Each experiment is self-contained and can be run independently. The visualizations are designed for paper figures.

Good luck with your research! 🚀
