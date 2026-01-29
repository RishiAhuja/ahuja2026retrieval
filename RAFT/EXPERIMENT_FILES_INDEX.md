# 📋 EXPERIMENT FILES INDEX

## Quick Reference: What to Run and Where

---

## ✅ CORRECTED EXPERIMENTS (USE THESE!)

### Location: `experiments_patchtst/corrected/`

| File | Purpose | Runtime | Priority |
|------|---------|---------|----------|
| **phase1_patchtst_336_validation.py** | Validate implementation | 4h | CRITICAL |
| **phase2a_patchtst_720_corrected.py** | Fair comparison at 720 | 3-4h | HIGH |
| **phase2b_patchtst_3000_corrected_CRITICAL.py** | THE critical test | 8-12h | HIGH |
| **run_all_corrected.sh** | Master script (runs all) | 20-24h | - |

**Usage:**
```bash
cd experiments_patchtst/corrected
bash run_all_corrected.sh
```

---

## ⛔ DEPRECATED EXPERIMENTS (DO NOT USE!)

### Location: `experiments_patchtst/`

| File | Why Deprecated | Use Instead |
|------|----------------|-------------|
| exp_patchtst_sl720.py | Wrong config (10 epochs, type1 LR, batch=16) | phase2a_patchtst_720_corrected.py |
| exp_patchtst_sl3000.py | Wrong config (10 epochs, type1 LR, batch=8) | phase2b_patchtst_3000_corrected_CRITICAL.py |

**Status:** Files updated with deprecation warnings

---

## 📊 ORIGINAL EXPERIMENT FILES (Reference Only)

### Location: `experiments/` and `final/code/`

These were used for Time-CAG experiments and RAFT baseline. They are CORRECT and don't need changes:

| File | Purpose | Status |
|------|---------|--------|
| exp1_timecag_v1.py | Time-CAG at seq_len=3000 | ✅ Correct |
| exp2_timecag_v2.py | Time-CAG at seq_len=1440 | ✅ Correct |
| exp3_timecag_v3.py | Time-CAG at seq_len=720 | ✅ Correct |
| exp4_ablation_dmodel.py | d_model ablation | ✅ Correct |
| exp5_ablation_elayers.py | e_layers ablation | ✅ Correct |
| exp6_ablation_dropout.py | dropout ablation | ✅ Correct |
| exp7_best_config.py | Best Time-CAG config | ✅ Correct |
| validate_etth2.py | ETTh2 validation | ✅ Correct |
| run_exchange_experiments.py | Exchange dataset | ✅ Correct |

**Note:** Only PatchTST experiments had config errors. All other experiments are fine.

---

## 📖 DOCUMENTATION FILES

| File | Purpose |
|------|---------|
| **QUICK_START.md** | Start here! Quick guide |
| **CRITICAL_HYPERPARAMETER_AUDIT.md** | Detailed error analysis |
| **SERVER_EXPERIMENT_PLAN.md** | Complete experimental protocol |
| **PAPER_VIABILITY_ASSESSMENT.md** | Outcome scenarios & paper framing |
| **EXPERIMENT_FILES_INDEX.md** | This file |

---

## 🗂️ DIRECTORY STRUCTURE

```
RAFT/
├── experiments_patchtst/
│   ├── corrected/                    ← NEW: Use these!
│   │   ├── phase1_patchtst_336_validation.py
│   │   ├── phase2a_patchtst_720_corrected.py
│   │   ├── phase2b_patchtst_3000_corrected_CRITICAL.py
│   │   ├── run_all_corrected.sh      ← Master script
│   │   └── results/                   ← Output directory
│   │
│   ├── exp_patchtst_sl720.py         ← DEPRECATED
│   └── exp_patchtst_sl3000.py        ← DEPRECATED
│
├── experiments/                       ← Original Time-CAG experiments (OK)
│   ├── exp1_timecag_v1.py
│   ├── exp2_timecag_v2.py
│   └── ... (all correct)
│
├── final/                             ← Organized results (OK)
│   ├── code/                          ← All experiment files
│   ├── results/                       ← All result files
│   └── FILE_MANIFEST.md               ← File documentation
│
├── QUICK_START.md                     ← START HERE
├── CRITICAL_HYPERPARAMETER_AUDIT.md   ← What went wrong
├── SERVER_EXPERIMENT_PLAN.md          ← What to run
├── PAPER_VIABILITY_ASSESSMENT.md      ← What happens next
└── EXPERIMENT_FILES_INDEX.md          ← This file
```

---

## 📝 CONFIGURATION COMPARISON

### PatchTST-720:

| Parameter | OLD (Wrong) | NEW (Correct) |
|-----------|-------------|---------------|
| File | exp_patchtst_sl720.py | phase2a_patchtst_720_corrected.py |
| train_epochs | 10 | 100 |
| batch_size | 16 | 128 |
| lradj | 'type1' | 'TST' |
| d_model | 128 | 16 |
| n_heads | 8 | 4 |
| d_ff | 256 | 128 |
| dropout | 0.1 | 0.3 |
| fc_dropout | (missing) | 0.3 |

### PatchTST-3000:

| Parameter | OLD (Wrong) | NEW (Correct) |
|-----------|-------------|---------------|
| File | exp_patchtst_sl3000.py | phase2b_patchtst_3000_corrected_CRITICAL.py |
| train_epochs | 10 | 100 |
| batch_size | 8 | 128 |
| lradj | 'type1' | 'TST' |
| d_model | 128 | 16 |
| n_heads | 8 | 4 |
| d_ff | 256 | 128 |
| dropout | 0.1 | 0.3 |
| fc_dropout | (missing) | 0.3 |

**Key Difference:** Batch size was HALVED in old 3000 config (8 vs 16), now it's CONSISTENT (128 for both)

---

## 🎯 WHAT TO RUN

### Minimum (Validation + Critical Test):
```bash
cd experiments_patchtst/corrected

# Phase 1: Validation (~4h)
python3 phase1_patchtst_336_validation.py

# Phase 2A: seq_len=720 (~4h)
python3 phase2a_patchtst_720_corrected.py

# Phase 2B: seq_len=3000 (~12h)
python3 phase2b_patchtst_3000_corrected_CRITICAL.py
```

**Total: ~20 hours**

### Automated (Recommended):
```bash
cd experiments_patchtst/corrected
bash run_all_corrected.sh
```

**Total: ~24 hours (includes error handling)**

---

## 📊 EXPECTED RESULTS

### Phase 1 (Validation):
- **Expected:** MSE ≈ 0.377-0.380
- **If passed:** Implementation correct, proceed to Phase 2
- **If failed:** Fix code before continuing

### Phase 2A (720):
- **Expected:** MSE ≈ 0.38-0.42
- **Compare:** RAFT-720 = 0.379

### Phase 2B (3000) - THE CRITICAL TEST:
- **Scenario A:** MSE > 0.50 → Your finding validated!
- **Scenario B:** MSE < 0.45 → Original paper correct
- **Scenario C:** 0.45 < MSE < 0.50 → Ambiguous

---

## 🔍 HOW TO CHECK RESULTS

### Command Line:
```bash
# Check validation status
cat experiments_patchtst/corrected/results/phase1_validation_336.json | grep validation_status

# Check Phase 2B scenario
cat experiments_patchtst/corrected/results/phase2b_patchtst_3000_corrected_CRITICAL.json | grep scenario

# View all MSE values
grep '"test_mse"' experiments_patchtst/corrected/results/*.json
```

### Python:
```python
import json

# Load Phase 2B results
with open('experiments_patchtst/corrected/results/phase2b_patchtst_3000_corrected_CRITICAL.json') as f:
    results = json.load(f)

print(f"Scenario: {results['scenario']}")
print(f"MSE: {results['test_mse']}")
print(f"Recommendation: {results['recommendation']}")
```

---

## ✅ FILE STATUS SUMMARY

| File Type | Count | Status |
|-----------|-------|--------|
| Corrected PatchTST experiments | 3 | ✅ Ready to run |
| Deprecated PatchTST experiments | 2 | ⛔ Do not use |
| Original Time-CAG experiments | 15 | ✅ Correct (no changes) |
| Master run script | 1 | ✅ Ready to use |
| Documentation files | 5 | ✅ Complete |
| **Total new/modified files** | **11** | **✅ All ready** |

---

## 🚀 QUICK ACTION CHECKLIST

- [ ] Read QUICK_START.md
- [ ] Read CRITICAL_HYPERPARAMETER_AUDIT.md
- [ ] Get GPU server access
- [ ] Upload code to server
- [ ] Run `bash experiments_patchtst/corrected/run_all_corrected.sh`
- [ ] Wait ~24 hours
- [ ] Check results/phase2b_patchtst_3000_corrected_CRITICAL.json
- [ ] Determine scenario (A, B, or C)
- [ ] Revise paper accordingly
- [ ] Submit!

---

**Last Updated:** 2026-01-29  
**Status:** All files ready for execution  
**Next Step:** Get GPU and run experiments
