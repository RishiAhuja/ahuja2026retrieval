# PatchTST Setup Instructions

## Step 1: Get PatchTST Implementation

### Option A: Time-Series-Library (Recommended)
```bash
cd /Users/rishi/StudioProjects/caft/RAFT
git clone https://github.com/thuml/Time-Series-Library.git temp_tsl

# Copy PatchTST model
cp temp_tsl/models/PatchTST.py ./models/

# Copy any dependencies
cp temp_tsl/layers/Embed.py ./layers/ 2>/dev/null || true
cp temp_tsl/layers/SelfAttention_Family.py ./layers/ 2>/dev/null || true

# Cleanup
rm -rf temp_tsl
```

### Option B: Original PatchTST Repository
```bash
cd /Users/rishi/StudioProjects/caft/RAFT
git clone https://github.com/yuqinie98/PatchTST.git temp_patchtst

cp temp_patchtst/PatchTST_supervised/models/PatchTST.py ./models/

rm -rf temp_patchtst
```

## Step 2: Update models/__init__.py

Add this import:
```python
from .PatchTST import Model as PatchTST

__all__ = ['RAFT', 'TransformerLongContext', 'PatchTST']
```

## Step 3: Verify Installation

Test if PatchTST loads:
```bash
python -c "from models import PatchTST; print('✅ PatchTST imported successfully')"
```

## Step 4: Create Experiment Files

Run this to generate experiment scripts:
```bash
cd experiments_patchtst
# We'll create these next
```

## Step 5: Test One Epoch

Before running full experiments, test 1 epoch to verify everything works:
```bash
python experiments_patchtst/test_patchtst_setup.py
```

## Expected File Structure After Setup

```
RAFT/
├── models/
│   ├── __init__.py          (updated)
│   ├── RAFT.py
│   ├── TransformerLongContext.py
│   └── PatchTST.py          (NEW)
├── layers/
│   ├── Embed.py             (if needed)
│   └── SelfAttention_Family.py (if needed)
├── experiments_patchtst/
│   ├── README.md
│   ├── setup_instructions.md
│   ├── experiment_plan.json
│   ├── exp_patchtst_sl720.py    (to create)
│   ├── exp_patchtst_sl3000.py   (to create)
│   └── results/
│       └── training_logs/
```

## Troubleshooting

### Import Error: "No module named 'layers.Embed'"
**Solution**: Copy Embed.py from Time-Series-Library

### Import Error: "No module named 'einops'"
**Solution**: 
```bash
pip install einops
```

### CUDA Out of Memory
**Solution**: Reduce batch_size in configs
- seq_len=720: batch_size=16
- seq_len=3000: batch_size=4 or 8

## Next Steps

Once setup is complete:
1. Create exp_patchtst_sl720.py
2. Create exp_patchtst_sl3000.py
3. Run experiments
4. Compare results
