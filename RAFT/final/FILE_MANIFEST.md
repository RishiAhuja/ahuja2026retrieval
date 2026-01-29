# Experiment Files Manifest

**Date Created:** 29 January 2026  
**Purpose:** Documentation of all experiment code and result files with their original paths

---

## Directory Structure

```
final/
├── code/          # Experiment Python scripts
├── results/       # Experiment result JSON files and logs
└── FILE_MANIFEST.md  # This file
```

---

## 📁 CODE FILES (final/code/)

### Main Run Scripts

| File Name | Original Path | Description |
|-----------|---------------|-------------|
| `run.py` | `/Users/rishi/StudioProjects/caft/RAFT/run.py` | Main RAFT model runner with full argument parser |
| `run_timecag.py` | `/Users/rishi/StudioProjects/caft/RAFT/run_timecag.py` | Time-CAG (TransformerLongContext) runner |

---

### ETTh1 Core Time-CAG Experiments

| File Name | Original Path | Description |
|-----------|---------------|-------------|
| `exp1_timecag_v1.py` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/exp1_timecag_v1.py` | Exp1: Time-CAG v1 (seq_len=3000, d_model=256, e_layers=3) |
| `exp2_timecag_v2.py` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/exp2_timecag_v2.py` | Exp2: Time-CAG v2 (seq_len=1440, d_model=128, e_layers=3) |
| `exp3_timecag_v3.py` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/exp3_timecag_v3.py` | Exp3: Time-CAG v3 (seq_len=720, d_model=128, e_layers=3) |
| `exp7_best_config.py` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/exp7_best_config.py` | Exp7: Best config from ablations (d_model=64, e_layers=2, dropout=0.2) |

---

### ETTh1 Ablation Studies

| File Name | Original Path | Description |
|-----------|---------------|-------------|
| `exp4_ablation_dmodel.py` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/exp4_ablation_dmodel.py` | Ablation: d_model ∈ {32, 64, 128, 256} |
| `exp5_ablation_elayers.py` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/exp5_ablation_elayers.py` | Ablation: e_layers ∈ {1, 2, 3, 4} |
| `exp6_ablation_dropout.py` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/exp6_ablation_dropout.py` | Ablation: dropout ∈ {0.0, 0.1, 0.2, 0.3} |

---

### PatchTST Experiments (ETTh1)

| File Name | Original Path | Description |
|-----------|---------------|-------------|
| `exp_patchtst_sl720.py` | `/Users/rishi/StudioProjects/caft/RAFT/experiments_patchtst/exp_patchtst_sl720.py` | PatchTST with seq_len=720 (baseline comparison) |
| `exp_patchtst_sl3000.py` | `/Users/rishi/StudioProjects/caft/RAFT/experiments_patchtst/exp_patchtst_sl3000.py` | PatchTST with seq_len=3000 (degradation test) |

---

### Cross-Domain Validation Experiments

| File Name | Original Path | Description |
|-----------|---------------|-------------|
| `validate_etth2.py` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/validate_etth2.py` | ETTh2 + Exchange cross-domain validation suite |
| `run_exchange_experiments.py` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/run_exchange_experiments.py` | Exchange dataset validation experiments |
| `run_patchtst_3000_etth2.py` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/run_patchtst_3000_etth2.py` | PatchTST 3000 on ETTh2 dataset |
| `run_patchtst_3000_exchange.py` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/run_patchtst_3000_exchange.py` | PatchTST 3000 on Exchange dataset |

---

## 📊 RESULT FILES (final/results/)

### ETTh1 Core Experiment Results

| File Name | Original Path | Experiment | MSE Result |
|-----------|---------------|------------|------------|
| `exp1_timecag_v1.json` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/results/1.json` | Time-CAG v1 (seq=3000) | 1.323 |
| `exp2_timecag_v2.json` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/results/2.json` | Time-CAG v2 (seq=1440) | 0.484 |
| `exp3_timecag_v3.json` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/results/3.json` | Time-CAG v3 (seq=720) | 0.441 |
| `exp7_best_config.json` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/results/7.json` | Best configuration | 0.416 |

---

### ETTh1 Ablation Study Results

| File Name | Original Path | Study | Best Value |
|-----------|---------------|-------|------------|
| `exp4_ablation_dmodel.json` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/results/4.json` | d_model ablation | d_model=32 best |
| `exp5_ablation_elayers.json` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/results/5.json` | e_layers ablation | e_layers=2 best |
| `exp6_ablation_dropout.json` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/results/6.json` | dropout ablation | dropout=0.1 best |

---

### PatchTST Results (ETTh1)

| File Name | Original Path | Configuration | MSE Result |
|-----------|---------------|---------------|------------|
| `patchtst_720.json` | `/Users/rishi/StudioProjects/caft/RAFT/experiments_patchtst/results/patch_720.json` | PatchTST seq_len=720 | 0.385 |
| `patchtst_3000.json` | `/Users/rishi/StudioProjects/caft/RAFT/experiments_patchtst/results/patch_3000.json` | PatchTST seq_len=3000 | 0.647 |

---

### Cross-Domain Validation Results

| File Name | Original Path | Description |
|-----------|---------------|-------------|
| `final_cross_domain_validation.json` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/results/final_cross_domain_validation.json` | Complete ETTh2 + Exchange validation results |
| `cross_domain.json` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/results/cross_domain.json` | Cross-domain summary |
| `exchange_validation.json` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/results/exchange_validation.json` | Exchange dataset specific results |
| `patchtst_3000_exchange.json` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/results/patchtst_3000_exchange.json` | PatchTST 3000 on Exchange |
| `pathtst_3000_ethh2.json` | `/Users/rishi/StudioProjects/caft/RAFT/experiments/results/pathtst_3000_ethh2.json` | PatchTST 3000 on ETTh2 |

---

### RAFT Baseline Results

| File Name | Original Path | Description |
|-----------|---------------|-------------|
| `raft_baseline_results.txt` | `/Users/rishi/StudioProjects/caft/RAFT/result_long_term_forecast.txt` | RAFT baseline results (MSE: 0.379 at seq_len=720) |

---

## 📈 Key Results Summary

### ETTh1 Dataset (Primary)

| Model | Seq Len | MSE | Result File |
|-------|---------|-----|-------------|
| **RAFT** | 720 | **0.379** | `raft_baseline_results.txt` |
| PatchTST | 720 | 0.385 | `patchtst_720.json` |
| Time-CAG Best | 720 | 0.416 | `exp7_best_config.json` |
| Time-CAG v3 | 720 | 0.441 | `exp3_timecag_v3.json` |
| Time-CAG v2 | 1440 | 0.484 | `exp2_timecag_v2.json` |
| PatchTST | 3000 | 0.647 | `patchtst_3000.json` |
| Time-CAG v1 | 3000 | 1.323 | `exp1_timecag_v1.json` |

**Key Finding:** PatchTST degrades 68% (0.385 → 0.647) when context increases from 720 to 3000

---

### ETTh2 Dataset (Cross-Domain Validation)

| Model | Seq Len | MSE | Source |
|-------|---------|-----|--------|
| RAFT | 720 | 0.2817 | `final_cross_domain_validation.json` |
| PatchTST | 720 | 0.3073 | `final_cross_domain_validation.json` |
| PatchTST | 3000 | 0.5327 | `final_cross_domain_validation.json` |

**Degradation:** 73.36% (confirms pattern generalizes)

---

### Exchange Dataset (Cross-Domain Validation)

| Model | Seq Len | MSE | Source |
|-------|---------|-----|--------|
| RAFT | 720 | 0.0907 | `final_cross_domain_validation.json` |
| PatchTST | 720 | 0.0928 | `final_cross_domain_validation.json` |
| PatchTST | 3000 | 0.3496 | `final_cross_domain_validation.json` |

**Degradation:** 276.77% (catastrophic - worse than electricity domain)

---

## 🔬 Experiment Categories

### Category 1: Context Length Analysis (Paper Results Section 4.2)
- `exp1_timecag_v1.py` → 3000 tokens
- `exp2_timecag_v2.py` → 1440 tokens  
- `exp3_timecag_v3.py` → 720 tokens

### Category 2: Ablation Studies (Paper Table 2 - if included)
- `exp4_ablation_dmodel.py` → d_model study
- `exp5_ablation_elayers.py` → e_layers study
- `exp6_ablation_dropout.py` → dropout study

### Category 3: Baseline Comparisons (Paper Results Section 4.1)
- RAFT baseline: 0.379 MSE
- PatchTST-720: 0.385 MSE
- PatchTST-3000: 0.647 MSE (68% degradation)

### Category 4: Cross-Domain Validation (Paper Results Section 4.3)
- `validate_etth2.py` → ETTh2 experiments
- `run_exchange_experiments.py` → Exchange experiments
- Proves degradation generalizes across domains

---

## 📝 Notes

1. **All files are COPIES** - originals remain in their source locations
2. Result JSON files have been renamed from numbered (1.json, 2.json, etc.) to descriptive names
3. Total experiments: 7 core + 3 ablations + 2 PatchTST + 4 cross-domain = **16 experiment files**
4. All results are from actual runs on ETTh1, ETTh2, and Exchange datasets
5. MSE values rounded to 3 decimal places in paper for readability

---

## 🎯 Paper Citation Map

When citing results in the paper, use these files as sources:

- **"RAFT achieves 0.379 MSE"** → `raft_baseline_results.txt`
- **"PatchTST degrades 68%"** → `patchtst_720.json` + `patchtst_3000.json`
- **"Context length analysis"** → `exp1_timecag_v1.json`, `exp2_timecag_v2.json`, `exp3_timecag_v3.json`
- **"Cross-domain validation"** → `final_cross_domain_validation.json`
- **"Ablation studies"** → `exp4_ablation_dmodel.json`, `exp5_ablation_elayers.json`, `exp6_ablation_dropout.json`

---

## 📧 Contact

For questions about specific experiments or results, refer to the original path and check the experiment's Python file for configuration details.

**Repository Root:** `/Users/rishi/StudioProjects/caft/RAFT/`

---

*End of Manifest*
