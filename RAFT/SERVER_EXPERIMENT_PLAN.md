# 🚀 SERVER EXPERIMENT PLAN - CORRECTED CONFIGURATIONS

**Generated:** 2026-01-29  
**Status:** READY FOR EXECUTION  
**Expected Runtime:** ~24-48 hours on GPU server

---

## 📋 EXECUTIVE SUMMARY

### What We're Doing:
Re-running **PatchTST experiments with CORRECT hyperparameters** from the original paper to validate:
1. Whether PatchTST actually degrades with longer context (our claim)
2. Or if our original results were due to configuration errors (critical feedback)

### Why This Matters:
- **Current claim:** "PatchTST degrades 68% when context increases 720→3000"
- **Critical feedback:** "Original PatchTST paper showed IMPROVEMENTS with longer context on same dataset"
- **Risk:** Our paper could be rejected for contradicting established results with flawed experiments

### Paper Viability:
✅ **YES - Paper is still viable IF:**
1. We re-run with correct configs and results STILL show degradation → **Novel finding**
2. We acknowledge configuration issues and reframe contribution → **Methodological insight**
3. We pivot to "RAFT outperforms even properly-tuned PatchTST" → **Stronger claim**

❌ **NO - Paper is NOT viable IF:**
- Corrected PatchTST matches original paper (improves with context)
- We have no unique contribution beyond reproducing known results

---

## 🎯 EXPERIMENTS TO RUN

### Phase 1: VALIDATION (Priority: CRITICAL)
**Goal:** Reproduce original PatchTST results to validate our setup

#### Experiment 1A: PatchTST-336 (Original Paper Baseline)
```python
# File: experiments_patchtst/corrected/patchtst_336_original.py
seq_len = 336
pred_len = 96
batch_size = 128
train_epochs = 100
lradj = 'TST'
pct_start = 0.3
d_model = 16      # ← ORIGINAL (not 128!)
n_heads = 4       # ← ORIGINAL (not 8!)
e_layers = 3
d_ff = 128        # ← ORIGINAL (not 256!)
dropout = 0.3     # ← ORIGINAL (not 0.1!)
fc_dropout = 0.3
head_dropout = 0
patch_len = 16
stride = 8
learning_rate = 0.0001
```

**Expected Result:** MSE ≈ 0.377-0.380 (match original paper Table 1)  
**If doesn't match:** Implementation bug - STOP and fix before proceeding

---

### Phase 2: FAIR COMPARISON (Priority: HIGH)
**Goal:** Compare PatchTST at different context lengths with SAME configuration

#### Experiment 2A: PatchTST-720 (Corrected)
```python
# File: experiments_patchtst/corrected/patchtst_720_corrected.py
seq_len = 720      # ← Our comparison point
pred_len = 96
batch_size = 128   # ← WAS 16 (8x too small!)
train_epochs = 100 # ← WAS 10 (10x too few!)
lradj = 'TST'      # ← WAS 'type1' (WRONG!)
pct_start = 0.3
d_model = 16       # ← WAS 128 (8x too large!)
n_heads = 4        # ← WAS 8
e_layers = 3
d_ff = 128         # ← WAS 256
dropout = 0.3      # ← WAS 0.1
fc_dropout = 0.3   # ← NEW (was missing!)
head_dropout = 0   # ← NEW (was missing!)
patch_len = 16
stride = 8
learning_rate = 0.0001
```

**Expected Result:** MSE ≈ 0.38-0.42 (similar to 336, maybe slightly worse)  
**Runtime:** ~3-4 hours

---

#### Experiment 2B: PatchTST-3000 (Corrected)
```python
# File: experiments_patchtst/corrected/patchtst_3000_corrected.py
seq_len = 3000     # ← Long context test
pred_len = 96
batch_size = 128   # ← WAS 8 (16x too small!)
train_epochs = 100 # ← WAS 10 (10x too few!)
lradj = 'TST'      # ← WAS 'type1' (WRONG!)
pct_start = 0.3
d_model = 16       # ← WAS 128
n_heads = 4        # ← WAS 8
e_layers = 3
d_ff = 128         # ← WAS 256
dropout = 0.3      # ← WAS 0.1
fc_dropout = 0.3   # ← NEW
head_dropout = 0   # ← NEW
patch_len = 16
stride = 8
learning_rate = 0.0001
```

**Expected Result (Two Scenarios):**

**Scenario A (Original Paper is Right):**
- MSE ≈ 0.35-0.39 (IMPROVES with longer context)
- ❌ Contradicts our claim → Paper needs major revision

**Scenario B (Our Finding is Right):**
- MSE > 0.50 (DEGRADES even with correct config)
- ✅ Novel finding → Paper is valid!

**Runtime:** ~8-12 hours

---

### Phase 3: CROSS-DOMAIN VALIDATION (Priority: MEDIUM)
**Goal:** Validate degradation pattern on other datasets

#### Experiment 3A: PatchTST-720 on ETTh2
```python
# File: experiments_patchtst/corrected/patchtst_720_etth2.py
data = 'ETTh2'
data_path = 'ETTh2.csv'
seq_len = 720
# ... rest same as Experiment 2A
```

**Runtime:** ~3 hours

---

#### Experiment 3B: PatchTST-3000 on ETTh2
```python
# File: experiments_patchtst/corrected/patchtst_3000_etth2.py
data = 'ETTh2'
data_path = 'ETTh2.csv'
seq_len = 3000
# ... rest same as Experiment 2B
```

**Runtime:** ~8 hours

---

#### Experiment 3C: PatchTST-720 on Exchange
```python
# File: experiments_patchtst/corrected/patchtst_720_exchange.py
data = 'custom'
data_path = 'exchange_rate.csv'
seq_len = 720
enc_in = 8  # Exchange has 8 features
# ... rest same as Experiment 2A
```

**Runtime:** ~3 hours

---

#### Experiment 3D: PatchTST-3000 on Exchange
```python
# File: experiments_patchtst/corrected/patchtst_3000_exchange.py
data = 'custom'
data_path = 'exchange_rate.csv'
seq_len = 3000
enc_in = 8
# ... rest same as Experiment 2B
```

**Runtime:** ~8 hours

---

### Phase 4: RAFT RE-VALIDATION (Priority: LOW)
**Goal:** Ensure RAFT results are reproducible

#### Experiment 4: RAFT-720 Replication
```python
# File: experiments/raft_720_revalidation.py
# Just re-run to confirm MSE = 0.379
```

**Runtime:** ~2 hours

---

## 📊 TOTAL EXPERIMENT COUNT

| Phase | Experiments | GPU Hours | Priority |
|-------|-------------|-----------|----------|
| Phase 1 | 1 (validation) | 4h | CRITICAL |
| Phase 2 | 2 (ETTh1 720/3000) | 16h | HIGH |
| Phase 3 | 4 (cross-domain) | 22h | MEDIUM |
| Phase 4 | 1 (RAFT check) | 2h | LOW |
| **TOTAL** | **8 experiments** | **~44h** | - |

**Recommendation:** Run Phases 1-2 first (20 hours), evaluate results, then decide on Phase 3.

---

## 🗂️ FILES TO EDIT/CREATE

### Files to CREATE (New corrected experiments):
1. `experiments_patchtst/corrected/patchtst_336_original.py` ← Phase 1
2. `experiments_patchtst/corrected/patchtst_720_corrected.py` ← Phase 2A
3. `experiments_patchtst/corrected/patchtst_3000_corrected.py` ← Phase 2B
4. `experiments_patchtst/corrected/patchtst_720_etth2.py` ← Phase 3A
5. `experiments_patchtst/corrected/patchtst_3000_etth2.py` ← Phase 3B
6. `experiments_patchtst/corrected/patchtst_720_exchange.py` ← Phase 3C
7. `experiments_patchtst/corrected/patchtst_3000_exchange.py` ← Phase 3D
8. `experiments_patchtst/corrected/run_all_corrected.sh` ← Master script

### Files to EDIT (Original files - deprecated):
1. `experiments_patchtst/exp_patchtst_sl720.py` ← Add deprecation notice
2. `experiments_patchtst/exp_patchtst_sl3000.py` ← Add deprecation notice

### Files to REFERENCE (Original PatchTST):
- GitHub: `yuqinie98/PatchTST/scripts/PatchTST/etth1.sh`
- Paper: "A Time Series is Worth 64 Words" (ICLR 2023)

---

## 🔧 KEY CONFIGURATION CHANGES

### What We Changed:

| Parameter | OLD (Wrong) | NEW (Correct) | Impact |
|-----------|-------------|---------------|--------|
| `train_epochs` | 10 | 100 | 10x more training |
| `lradj` | 'type1' | 'TST' | Proper LR schedule |
| `batch_size` (720) | 16 | 128 | 8x larger batches |
| `batch_size` (3000) | 8 | 128 | 16x larger batches |
| `d_model` | 128 | 16 | 8x smaller model |
| `n_heads` | 8 | 4 | 2x fewer heads |
| `d_ff` | 256 | 128 | 2x smaller FFN |
| `dropout` | 0.1 | 0.3 | 3x more dropout |
| `fc_dropout` | (missing) | 0.3 | Added regularization |
| `head_dropout` | (missing) | 0 | Added parameter |

### Why These Changes Matter:

**Old Config Problems:**
- LR decayed to ~0.0000002 by epoch 10 → model stopped learning
- Only 10 epochs → severely undertrained (needs 100)
- Batch size halved for 3000 → unfair comparison + slower convergence
- Model 8x larger → needs more training, got less

**New Config Benefits:**
- OneCycle LR: warmup + proper decay over 100 epochs
- Consistent batch size → fair comparison
- Smaller model → faster training, better generalization
- More dropout → better regularization for long context

---

## 💻 SERVER REQUIREMENTS

### Hardware:
- **GPU:** 1x NVIDIA A100 (40GB) or V100 (32GB)
- **RAM:** 64GB minimum
- **Storage:** 50GB free space
- **CPU:** 16+ cores recommended

### Software:
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+
- See `requirements.txt` for full dependencies

### Estimated Costs:
- AWS p3.2xlarge (V100): ~$3/hour × 44 hours = **$132**
- AWS p4d.24xlarge (A100): ~$32/hour × 20 hours = **$640** (parallel)
- Google Cloud TPU v3: ~$8/hour × 30 hours = **$240**

**Budget Recommendation:** $150-300 for serial execution on V100

---

## 📈 SUCCESS CRITERIA

### Phase 1 (Validation):
- ✅ **PASS:** PatchTST-336 MSE = 0.377-0.380 (within 1% of original paper)
- ❌ **FAIL:** MSE differs by >5% → Implementation bug, fix before continuing

### Phase 2 (Critical Test):
- ✅ **Scenario A:** PatchTST-3000 MSE < PatchTST-720 MSE → Original paper correct
- ✅ **Scenario B:** PatchTST-3000 MSE > 0.50 → Our finding validated
- ❌ **Scenario C:** Results unstable/contradictory → More experiments needed

### Phase 3 (Robustness):
- ✅ **Strong Finding:** Degradation pattern holds on 2+ datasets
- ⚠️ **Mixed Results:** Degradation dataset-specific
- ❌ **Weak Finding:** No consistent pattern

---

## 📝 PAPER REVISION SCENARIOS

### Scenario A: Our Finding is CORRECT (PatchTST degrades)
**Paper Angle:** "Even State-of-the-Art Patching Fails at Extreme Context"

**Key Claims:**
1. PatchTST degrades X% when context increases 720→3000 (with CORRECT config)
2. RAFT outperforms PatchTST by Y% at long context
3. Retrieval > Patching for extreme long-range dependencies

**Contributions:**
- Systematic study of context length scaling limits
- First to show PatchTST degradation at seq_len > 2000
- RAFT as superior alternative for extreme context

**Paper Viability:** ✅ **STRONG** - Novel, impactful finding

---

### Scenario B: Original Paper is CORRECT (PatchTST improves)
**Paper Angle:** "RAFT: Retrieval-Augmented Approach Outperforms Patching"

**Key Claims:**
1. Both PatchTST and RAFT benefit from longer context
2. RAFT shows Z% improvement over PatchTST across all context lengths
3. Retrieval provides complementary benefits to patching

**Contributions:**
- RAFT as orthogonal approach to PatchTST
- Retrieval-augmentation as general technique
- Comprehensive comparison on multiple datasets

**Paper Viability:** ⚠️ **MODERATE** - Less novel, but still valid

---

### Scenario C: Mixed Results
**Paper Angle:** "When Does Patching Fail? A Study of Context Length Limits"

**Key Claims:**
1. PatchTST scaling is dataset-dependent
2. RAFT provides robust performance across datasets
3. Retrieval-augmentation as safety mechanism

**Contributions:**
- Characterization of PatchTST failure modes
- Dataset-specific scaling analysis
- RAFT as robust baseline

**Paper Viability:** ⚠️ **MODERATE-WEAK** - Needs strong analysis

---

## 🚦 GO/NO-GO DECISION POINTS

### After Phase 1:
- ✅ **GO:** Validation passes → Continue to Phase 2
- ❌ **NO-GO:** Validation fails → Fix implementation, restart

### After Phase 2:
- ✅ **GO (Scenario A):** Degradation confirmed → Write Scenario A paper, run Phase 3 for robustness
- ✅ **GO (Scenario B):** Improvement confirmed → Write Scenario B paper, run Phase 3 for comparison
- ⚠️ **MAYBE:** Unclear results → Run Phase 3 for clarity

### After Phase 3:
- ✅ **SUBMIT:** Consistent findings across datasets
- ⚠️ **REVISE:** Mixed findings, need deeper analysis
- ❌ **PIVOT:** Contradictory findings, major revision needed

---

## 🎯 RECOMMENDED EXECUTION PLAN

### Week 1 (Validation + Critical Test):
1. **Day 1:** Set up server, install dependencies
2. **Day 2:** Run Phase 1 (Experiment 1A) - 4 hours
3. **Day 2-3:** Analyze Phase 1, verify matches original paper
4. **Day 3-4:** Run Phase 2 (Experiments 2A, 2B) - 16 hours
5. **Day 5:** Analyze Phase 2, determine paper scenario
6. **Day 6-7:** Initial paper draft based on results

### Week 2 (Cross-Domain + Finalization):
1. **Day 8-10:** Run Phase 3 (Experiments 3A-D) - 22 hours
2. **Day 11:** Consolidate all results
3. **Day 12-13:** Revise paper with complete findings
4. **Day 14:** Final review and submission prep

**Total Timeline:** 14 days (2 weeks)

---

## ✅ FINAL CHECKLIST

Before starting server runs:

- [ ] Read CRITICAL_HYPERPARAMETER_AUDIT.md
- [ ] Create all 8 experiment files with corrected configs
- [ ] Verify data files are present (ETTh1.csv, ETTh2.csv, exchange_rate.csv)
- [ ] Test one experiment locally to verify code runs
- [ ] Set up GPU server with required dependencies
- [ ] Create results directory structure
- [ ] Prepare logging/monitoring scripts
- [ ] Schedule regular result backups
- [ ] Set budget alerts for cloud costs
- [ ] Prepare paper template for quick revision

---

## 📞 SUPPORT

If experiments fail or results are unclear:
1. Check logs in `experiments_patchtst/corrected/logs/`
2. Compare training curves to original PatchTST paper
3. Verify data loading with `test_data_loader.py`
4. Check GPU memory usage (should be <30GB for batch_size=128)

---

**Last Updated:** 2026-01-29  
**Version:** 1.0  
**Status:** Ready for execution
