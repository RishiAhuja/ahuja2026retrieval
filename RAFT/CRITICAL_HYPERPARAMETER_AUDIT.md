# CRITICAL HYPERPARAMETER AUDIT - PatchTST Configuration Issues

## ⚠️ URGENT: Configuration Errors Identified

### Issue Summary
Your PatchTST experiments contradict the original PatchTST paper (Nie et al., 2023) which showed **performance improvements** with longer context windows on ETTh1. Your results show **68% degradation**. This audit identifies critical configuration mismatches.

---

## 🔴 CRITICAL DISCREPANCIES

### 1. **Training Epochs: SEVERELY INSUFFICIENT**
**Your Configuration:**
- PatchTST-720: `train_epochs = 10`
- PatchTST-3000: `train_epochs = 10`

**Original PatchTST Paper (ETTh1):**
- `train_epochs = 100`
- Early stopping with `patience = 3-20` (depends on dataset)

**Impact:** ❌ **CRITICAL** - Training for only 10 epochs vs 100 epochs means your model is **severely undertrained**. PatchTST requires many epochs to learn effective patch representations.

---

### 2. **Learning Rate Schedule: WRONG TYPE**
**Your Configuration:**
```python
lradj = 'type1'
learning_rate = 0.0001
```

**What `type1` does** (from original code):
```python
# type1: Aggressive exponential decay
lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
# Epoch 1: 0.0001
# Epoch 2: 0.00005
# Epoch 3: 0.000025
# Epoch 4: 0.0000125
# By epoch 10: lr ≈ 0.000000195 (essentially zero!)
```

**Original PatchTST Paper (ETTh1):**
```bash
lradj = 'TST'  # OneCycle learning rate scheduler
pct_start = 0.3
```

**Impact:** ❌ **CRITICAL** - Your learning rate decays to near-zero by epoch 10. With longer sequences (3000), the model barely trains because:
1. Longer sequences → fewer batches per epoch
2. Fewer batches + decaying LR → minimal weight updates
3. Result: **model cannot learn from longer context**

---

### 3. **Batch Size Reduction: UNFAIR COMPARISON**
**Your Configuration:**
- PatchTST-720: `batch_size = 16`
- PatchTST-3000: `batch_size = 8` (reduced by 50%)

**Original PatchTST Paper (ETTh1):**
- PatchTST (multivariate): `batch_size = 128`
- PatchTST (univariate): `batch_size = 128`

**Impact:** ⚠️ **HIGH** - Reducing batch size introduces multiple confounds:
1. Smaller batches → noisier gradients → slower convergence
2. Batch size reduction + fewer epochs = compounded undertraining
3. Unfair comparison: 720 gets 2x batch size advantage
4. With `type1` LR schedule + small batch, 3000-step model gets inadequate training

---

### 4. **Model Architecture: DIFFERENT FROM PAPER**
**Your Configuration:**
```python
d_model = 128
n_heads = 8
e_layers = 3
d_ff = 256
dropout = 0.1
```

**Original PatchTST Paper (ETTh1 multivariate):**
```bash
d_model = 16    # Much smaller!
n_heads = 4
e_layers = 3    # ✓ Correct
d_ff = 128
dropout = 0.3
fc_dropout = 0.3
head_dropout = 0
```

**Impact:** ⚠️ **MEDIUM** - Your model is **significantly larger** (128 vs 16 d_model):
- More parameters → requires more training epochs
- Larger model + 10 epochs + wrong LR schedule = undertraining
- Original used dropout=0.3, you used 0.1 (less regularization)

---

### 5. **Model Head Configuration: MISSING PARAMETERS**
**Your Configuration:**
- No explicit `fc_dropout`, `head_dropout` parameters

**Original PatchTST Paper:**
```bash
fc_dropout = 0.2-0.3
head_dropout = 0
```

**Impact:** ⚠️ **LOW-MEDIUM** - Missing regularization on prediction head

---

## 📊 WHY YOUR RESULTS CONTRADICT THE ORIGINAL PAPER

### Root Cause Analysis:

1. **Severe Undertraining (10 vs 100 epochs):**
   - PatchTST with seq_len=3000 has **4.17x more input tokens** than seq_len=720
   - Same number of timesteps → same number of batches per epoch
   - But 4x more tokens per sample → requires more epochs to converge
   - **Your 10 epochs is catastrophically insufficient**

2. **Learning Rate Collapse:**
   - `type1` schedule decays LR by 50% every epoch
   - By epoch 10, LR ≈ 0.000000195 (1000x smaller than initial)
   - Model with longer context needs more gradient updates
   - **Essentially stops learning after epoch 5**

3. **Batch Size Confound:**
   - seq_len=3000 uses batch_size=8 (half of 720's batch_size=16)
   - Combined with collapsing LR → even fewer effective weight updates
   - **Unfair comparison masked as memory constraint**

4. **Architecture Mismatch:**
   - Your d_model=128 is 8x larger than paper's d_model=16
   - Larger model + insufficient training = poor generalization
   - **Overfitted to training set, failed on test set**

### Expected Behavior vs Your Results:

| Aspect | Original PatchTST Paper | Your Experiment |
|--------|------------------------|-----------------|
| Longer context | ✅ Improves MSE | ❌ 68% degradation |
| Training | 100 epochs, OneCycle LR | 10 epochs, exponential decay |
| Convergence | Fully converged | Severely undertrained |
| Batch size | Consistent (128) | Inconsistent (16→8) |
| d_model | 16 (small, efficient) | 128 (large, undertrained) |

---

## 🔧 REQUIRED FIXES

### Fix #1: Training Duration
```python
train_epochs = 100  # NOT 10!
patience = 10       # NOT 3
```

### Fix #2: Learning Rate Schedule
```python
lradj = 'TST'      # OneCycle scheduler (NOT 'type1')
pct_start = 0.3    # Warmup for 30% of training
```

### Fix #3: Batch Size Consistency
```python
# Keep consistent or use gradient accumulation
batch_size = 16  # Same for both 720 and 3000
# OR if memory constrained:
# Use gradient accumulation to simulate batch_size=16 with smaller batches
```

### Fix #4: Model Architecture (Match Original Paper)
```python
# Option A: Use original paper config
d_model = 16
n_heads = 4
d_ff = 128
dropout = 0.3
fc_dropout = 0.3
head_dropout = 0

# Option B: If using larger model, train MUCH longer
d_model = 128
train_epochs = 200  # Needs 2x epochs due to larger capacity
```

### Fix #5: Add Missing Regularization
```python
fc_dropout = 0.3
head_dropout = 0
```

---

## 🚨 EXPERIMENTS TO RE-RUN (Priority Order)

### **Priority 1: URGENT - Validate Original Paper Claims**
Re-run PatchTST with **exact original configurations**:

```bash
# PatchTST-336 (original paper baseline)
seq_len = 336
pred_len = 96
batch_size = 128
train_epochs = 100
lradj = 'TST'
pct_start = 0.3
d_model = 16
n_heads = 4
d_ff = 128
dropout = 0.3
fc_dropout = 0.3

# If this does NOT match original paper results → implementation bug
# If this matches → proceed to Priority 2
```

### **Priority 2: HIGH - Fair Comparison at seq_len=720**
```bash
seq_len = 720
batch_size = 128
train_epochs = 100
lradj = 'TST'
d_model = 16  # Match original
# ... rest same as Priority 1
```

### **Priority 3: HIGH - Fair Comparison at seq_len=3000**
```bash
seq_len = 3000
batch_size = 128  # SAME as 720 (or use gradient accumulation)
train_epochs = 100
lradj = 'TST'
d_model = 16
# ... rest same as Priority 1
```

### **Priority 4: MEDIUM - Larger Model (if validated)**
Only if Priority 1-3 match original paper:
```bash
# Try larger model with proper training
d_model = 128
train_epochs = 200  # Compensate for larger capacity
lradj = 'TST'
batch_size = 128
```

---

## 📝 ADDITIONAL VALIDATION CHECKS

### Before Re-running:

1. **Verify Data Loading:**
   ```python
   # Check that data splits match original paper
   # ETTh1: 12 months train, 4 val, 4 test
   # Total: 17,420 rows → 8,640 / 2,880 / 2,880
   ```

2. **Verify Patch Calculation:**
   ```python
   # seq_len=336: patches = (336-16)/8 + 2 = 42 patches
   # seq_len=720: patches = (720-16)/8 + 2 = 90 patches
   # seq_len=3000: patches = (3000-16)/8 + 2 = 375 patches
   ```

3. **Monitor Training:**
   - Log validation loss every epoch
   - Check if loss is still decreasing at epoch 10 (it should be!)
   - Compare early stopping point between 720 and 3000

4. **Reproduce Original Paper Table:**
   - Get exact numbers from PatchTST paper Table 1
   - Your re-run should match within ±5%

---

## 🎯 CONCLUSION

### Current Status: ❌ **INVALID EXPERIMENTAL RESULTS**

Your experiments suffer from:
1. **90% reduction in training epochs** (10 vs 100)
2. **Wrong learning rate scheduler** (exponential decay vs OneCycle)
3. **Unfair batch size reduction** (8 vs 16)
4. **Architecture mismatch** (d_model 128 vs 16)

### These errors **completely invalidate** your claim that "PatchTST degrades with longer context."

### What Actually Happened:
- PatchTST-720: Undertrained but "lucky" (simpler task)
- PatchTST-3000: Undertrained AND insufficient LR (impossible task)
- Result: Artificial degradation due to **experimental design flaws**, not model limitations

### Recommendation:
1. ✅ **Immediately re-run experiments with corrected configurations**
2. ✅ **Use original PatchTST hyperparameters from their ETTh1 scripts**
3. ✅ **Train for 100 epochs minimum**
4. ✅ **Use 'TST' (OneCycle) learning rate schedule**
5. ✅ **Keep batch size consistent**
6. ⚠️ **Do NOT submit paper until validating against original results**

---

## 📚 Reference: Original PatchTST Configuration

From `yuqinie98/PatchTST` GitHub repository:

**File:** `scripts/PatchTST/etth1.sh`

```bash
seq_len=336
model_name=PatchTST

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3 \
      --fc_dropout 0.3 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --train_epochs 100 \
      --patience 20 \
      --lradj 'TST' \
      --pct_start 0.3 \
      --batch_size 128 \
      --learning_rate 0.0001
done
```

**This is what you should have used.**

---

Generated: 2024-01-XX
Status: CRITICAL - REQUIRES IMMEDIATE ACTION
