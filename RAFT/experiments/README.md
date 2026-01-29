# Time-CAG Experiments for Workshop Paper

This directory contains all 7 experiments designed to generate real data for the workshop paper comparing Time-CAG (continuous long context) vs RAFT (retrieval-augmented forecasting).

## Experiment Overview

| Exp | Name | seq_len | d_model | e_layers | dropout | Purpose |
|-----|------|---------|---------|----------|---------|---------|
| 1 | Time-CAG v1 | 3000 | 128 | 3 | 0.1 | Max context baseline |
| 2 | Time-CAG v2 | 1440 | 128 | 3 | 0.1 | Medium context (2x RAFT) |
| 3 | Time-CAG v3 | 720 | 128 | 3 | 0.1 | **Fair comparison** (same as RAFT) |
| 4 | Ablation d_model | 720 | **32,64,128,256** | 2 | 0.1 | Find optimal capacity |
| 5 | Ablation e_layers | 720 | 128 | **1,2,3,4** | 0.1 | Find optimal depth |
| 6 | Ablation dropout | 720 | 128 | 2 | **0.0,0.1,0.2,0.3** | Find optimal regularization |
| 7 | Best Config | 720 | 64 | 2 | 0.2 | **Final optimized model** |

## Running Experiments

### Option 1: Run All (Overnight)
```bash
cd /Users/rishi/StudioProjects/caft/RAFT/experiments
source ../venv/bin/activate
chmod +x run_all.sh
./run_all.sh
```

**Estimated time:** 8-12 hours

### Option 2: Run Individual Experiments
```bash
cd /Users/rishi/StudioProjects/caft/RAFT/experiments
source ../venv/bin/activate

# Run specific experiment
python exp1_timecag_v1.py
python exp2_timecag_v2.py
python exp3_timecag_v3.py
# etc.
```

### Option 3: Quick Test (1 epoch each)
Modify `train_epochs = 1` in each file to verify setup before full runs.

## Output Files

After running, you'll get:

```
experiments/
├── results/
│   ├── exp1_results.json    # MSE, MAE, training time
│   ├── exp2_results.json
│   ├── exp3_results.json
│   ├── exp4_results.json    # Ablation: d_model grid
│   ├── exp5_results.json    # Ablation: e_layers grid
│   ├── exp6_results.json    # Ablation: dropout grid
│   ├── exp7_results.json    # Best config final result
│   ├── exp1_log.txt         # Full training logs
│   ├── exp2_log.txt
│   └── ...
```

## Using Results for Graphs

After experiments complete:

1. **Update graph generation script:**
   ```bash
   python update_graphs.py  # Reads from results/*.json
   ```

2. **Generate final graphs:**
   ```bash
   python generate_graphs.py  # Uses real data now!
   ```

3. **Check outputs:**
   - `graph1_mse_comparison.png` - RAFT vs all Time-CAG variants
   - `graph2_loss_curves.png` - Training curves showing overfitting
   - `graph3_context_vs_mse.png` - More context ≠ better performance
   - `graph4_predictions.png` - Visual comparison of forecasts
   - `graph5_computational_cost.png` - Runtime vs accuracy
   - `graph6_ablation_heatmap.png` - Hyperparameter grid search

## RAFT Baseline

For comparison, RAFT's result:
- **MSE: 0.379** (from `run.py --data ETTh1`)
- Context length: 720
- Uses retrieval augmentation

**Goal:** See if any Time-CAG configuration can beat 0.379

## Notes

- All experiments use ETTh1 dataset
- Prediction length fixed at 96 timesteps
- Patching enabled (patch_size=12) like PatchTST
- MPS (Mac GPU) acceleration enabled
- Results automatically saved as JSON
- Training logs captured for debugging

## Expected Outcomes

Based on preliminary tests:
- ❌ Longer context (3000) likely worse than shorter (720)
- ❌ Time-CAG likely won't beat RAFT's 0.379
- ✅ Results will support "Retrieval > Long Context" thesis
- ✅ Valid negative results for workshop paper

## After Experiments

Update `generate_graphs.py` placeholders with actual values from `results/*.json` files, or use the automated `update_graphs.py` script.
