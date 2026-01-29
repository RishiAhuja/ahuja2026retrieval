# Colab Setup Instructions

## Step 1: Environment Setup
Paste this in Colab Cell 1:

```python
# ═══════════════════════════════════════════════════════════════
# CELL 1: Environment Setup
# ═══════════════════════════════════════════════════════════════
import os

# Clean start - remove any previous RAFT clones
!rm -rf /content/RAFT

# Install dependencies
!pip install -q torch torchvision sktime

# Clone RAFT repository
!git clone https://github.com/archon159/RAFT.git
os.chdir('/content/RAFT')

# Download ETT datasets
!mkdir -p data/ETT
!wget -q -O data/ETT/ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
!wget -q -O data/ETT/ETTh2.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv
!wget -q -O data/ETT/ETTm1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv
!wget -q -O data/ETT/ETTm2.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv

# Check GPU
!nvidia-smi --query-gpu=name --format=csv,noheader

print("\n✅ Environment setup complete!")
print(f"Working directory: {os.getcwd()}")
```

## Step 2: Upload Experiments
Paste this in Colab Cell 2:

```python
# ═══════════════════════════════════════════════════════════════
# CELL 2: Upload Experiments
# ═══════════════════════════════════════════════════════════════
import os
from google.colab import files

# Make sure we're in the right directory
os.chdir('/content/RAFT')
print(f"Current directory: {os.getcwd()}")

# Upload experiments.zip from your Mac
print("\n📤 Upload experiments.zip from:")
print("   /Users/rishi/StudioProjects/caft/RAFT/experiments.zip")
print("")

uploaded = files.upload()

# Extract
!unzip -o experiments.zip

# Verify extraction
print("\n🔍 Verifying files...")
!ls -la experiments/

expected_files = [
    'exp1_timecag_v1.py',
    'exp2_timecag_v2.py', 
    'exp3_timecag_v3.py',
    'exp4_ablation_dmodel.py',
    'exp5_ablation_elayers.py',
    'exp6_ablation_dropout.py',
    'exp7_best_config.py',
    'run_all.sh'
]

missing = []
for f in expected_files:
    if os.path.exists(f'experiments/{f}'):
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ MISSING: {f}")
        missing.append(f)

if missing:
    print(f"\n❌ {len(missing)} files missing!")
else:
    print("\n✅ All files uploaded successfully!")
    print("\nReady to run experiments!")
```

## Step 3: Run Experiments
Paste this in Colab Cell 3:

```python
# ═══════════════════════════════════════════════════════════════
# CELL 3: Run All Experiments (8-12 hours)
# ═══════════════════════════════════════════════════════════════
import os
os.chdir('/content/RAFT')

# Make script executable
!chmod +x experiments/run_all.sh

# Run all experiments (non-interactive mode)
!bash experiments/run_all.sh < /dev/null
```

## Step 4: Download Results
After completion, paste this in Colab Cell 4:

```python
# ═══════════════════════════════════════════════════════════════
# CELL 4: Download Results
# ═══════════════════════════════════════════════════════════════
from google.colab import files
import os

os.chdir('/content/RAFT')

# Create results archive
!zip -r results.zip experiments/results/

# Download
if os.path.exists('results.zip'):
    print('📥 Downloading results.zip...')
    files.download('results.zip')
    print('✅ Download complete!')
else:
    print('❌ results.zip not found')
```

## Troubleshooting

If you see nested RAFT directories:
```python
# Run this to check your location
!pwd
!ls -la
```

If experiments folder is missing:
```python
# Check where files were extracted
!find /content -name "exp1_timecag_v1.py" 2>/dev/null
```

## File Location Reference

- Mac location: `/Users/rishi/StudioProjects/caft/RAFT/experiments.zip`
- Colab after clone: `/content/RAFT/`
- After unzip: `/content/RAFT/experiments/exp1_timecag_v1.py`
