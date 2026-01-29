# 🚀 QUICK START GUIDE - Server Experiments

## What You Need to Do Right Now:

### Step 1: Read These Documents (30 mins)
1. [CRITICAL_HYPERPARAMETER_AUDIT.md](CRITICAL_HYPERPARAMETER_AUDIT.md) - What went wrong
2. [SERVER_EXPERIMENT_PLAN.md](SERVER_EXPERIMENT_PLAN.md) - What to run
3. [PAPER_VIABILITY_ASSESSMENT.md](PAPER_VIABILITY_ASSESSMENT.md) - What happens next

### Step 2: Get a GPU Server (1 hour)
**Recommended:** Lambda Labs (cheapest, ~$16 total)
- Visit: https://lambdalabs.com/
- Select: 1x V100 or A6000 GPU
- Region: Closest to you
- Instance type: On-demand (can stop anytime)

**Alternative:** University GPU cluster (free!)

### Step 3: Setup Server (30 mins)
```bash
# SSH into server
ssh username@server-ip

# Clone your repo
git clone [your-repo-url]
cd RAFT

# Install dependencies
pip install -r requirements.txt

# Verify GPU
nvidia-smi
```

### Step 4: Run Experiments (~24 hours)
```bash
# Navigate to corrected experiments
cd experiments_patchtst/corrected

# Run ALL experiments (automated)
bash run_all_corrected.sh

# OR run individual phases:
# bash run_all_corrected.sh 1   # Validation only (~4h)
# bash run_all_corrected.sh 2   # Critical test (~16h)
```

### Step 5: Analyze Results (1 hour)
```bash
# Check results
cat results/phase2b_patchtst_3000_corrected_CRITICAL.json

# Look for "scenario" field:
# - "OUR_FINDING_VALIDATED" → Your paper is VALID!
# - "ORIGINAL_PAPER_CORRECT" → Need major revision
# - "AMBIGUOUS" → Run more experiments
```

---

## Files Created for You:

### Experiment Files:
- ✅ `experiments_patchtst/corrected/phase1_patchtst_336_validation.py`
- ✅ `experiments_patchtst/corrected/phase2a_patchtst_720_corrected.py`
- ✅ `experiments_patchtst/corrected/phase2b_patchtst_3000_corrected_CRITICAL.py`
- ✅ `experiments_patchtst/corrected/run_all_corrected.sh` ← **Master script**

### Documentation:
- ✅ `CRITICAL_HYPERPARAMETER_AUDIT.md` - Error analysis
- ✅ `SERVER_EXPERIMENT_PLAN.md` - Detailed plan
- ✅ `PAPER_VIABILITY_ASSESSMENT.md` - Outcome scenarios

### Old Files (DO NOT USE):
- ⛔ `experiments_patchtst/exp_patchtst_sl720.py` - DEPRECATED
- ⛔ `experiments_patchtst/exp_patchtst_sl3000.py` - DEPRECATED

---

## Expected Timeline:

| Day | Task | Hours |
|-----|------|-------|
| Day 1 | Setup server | 2h |
| Day 2 | Run Phase 1 (validation) | 4h |
| Day 3 | Analyze Phase 1, start Phase 2A | 4h |
| Day 4 | Finish Phase 2A, start Phase 2B | 12h |
| Day 5 | Finish Phase 2B, analyze results | 4h |
| **Total** | | **~26h** |

---

## Cost Breakdown:

**Lambda Labs A6000:**
- $0.80/hour × 24 hours = **$19.20**

**AWS p3.2xlarge (V100):**
- $3.06/hour × 24 hours = **$73.44**

**Google Cloud V100:**
- $2.48/hour × 24 hours = **$59.52**

**University GPU:**
- **FREE** (if available)

---

## Decision Points:

### After Phase 1 (4 hours):
- ✅ **Validation MSE ≈ 0.38** → Continue
- ❌ **Validation MSE > 0.40** → Fix implementation, restart

### After Phase 2B (20 hours):
- ✅ **MSE > 0.50** → Your finding validated! Write paper.
- ⚠️ **MSE < 0.45** → Major revision needed. Reframe paper.
- ⚠️ **0.45 < MSE < 0.50** → Run Phase 3 (cross-domain)

---

## Three Possible Papers:

### Paper A: "PatchTST Fails at Extreme Context"
- **If:** MSE > 0.50 at seq_len=3000
- **Venue:** ICLR, NeurIPS (top-tier)
- **Contribution:** Novel finding

### Paper B: "RAFT Outperforms PatchTST"
- **If:** MSE < 0.45 at seq_len=3000
- **Venue:** ICML, AISTATS (strong)
- **Contribution:** Comprehensive comparison

### Paper C: "Scaling Analysis of Patch-Based Methods"
- **If:** Ambiguous results
- **Venue:** TMLR, AISTATS (solid)
- **Contribution:** Empirical analysis

**All three are publishable!**

---

## Need Help?

1. **Check logs:** `experiments_patchtst/corrected/logs/`
2. **Review results:** `experiments_patchtst/corrected/results/`
3. **Read audit:** `CRITICAL_HYPERPARAMETER_AUDIT.md`

---

## Bottom Line:

✅ **Your paper CAN be published**  
✅ **All experiment code is ready**  
✅ **Just need to run on GPU server**  
✅ **24 hours of compute → clear answer**  

**Next step:** Get GPU access and run `bash run_all_corrected.sh`

Good luck! 🚀
