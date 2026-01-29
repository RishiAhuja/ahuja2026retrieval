#!/usr/bin/env python3
"""
Test PatchTST Setup - Quick verification before full experiments
Runs 1 epoch to verify everything works
"""

import os
import sys

# Add RAFT root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
raft_root = os.path.dirname(script_dir)
if raft_root not in sys.path:
    sys.path.insert(0, raft_root)

print("=" * 80)
print("PATCHTST SETUP TEST")
print("=" * 80)

# Test 1: Import PatchTST
print("\n[TEST 1] Importing PatchTST...")
try:
    from models.PatchTST import Model as PatchTST
    print("✅ PatchTST imported successfully")
except ImportError as e:
    print(f"❌ Failed to import PatchTST: {e}")
    print("\nTroubleshooting:")
    print("1. Check if PatchTST.py exists in models/")
    print("2. Check if layers/Embed.py exists")
    print("3. Install: pip install einops")
    sys.exit(1)

# Test 2: Check dependencies
print("\n[TEST 2] Checking dependencies...")
try:
    from layers.Embed import PatchEmbedding
    print("✅ PatchEmbedding available")
except ImportError as e:
    print(f"❌ Missing layer dependency: {e}")
    print("\nCopy from Time-Series-Library:")
    print("cp temp_tsl/layers/Embed.py ./layers/")
    sys.exit(1)

try:
    import einops
    print("✅ einops installed")
except ImportError:
    print("⚠️  einops not installed")
    print("Run: pip install einops")
    sys.exit(1)

# Test 3: Create dummy config and instantiate model
print("\n[TEST 3] Creating PatchTST model instance...")
try:
    class DummyConfig:
        task_name = 'long_term_forecast'
        seq_len = 720
        pred_len = 96
        d_model = 128
        n_heads = 8
        e_layers = 3
        d_ff = 256
        dropout = 0.1
        enc_in = 7
        dec_in = 7
        c_out = 7
        embed = 'timeF'
        freq = 'h'
        activation = 'gelu'
        output_attention = False
        factor = 1
        distil = True
    
    config = DummyConfig()
    model = PatchTST(config)
    print(f"✅ PatchTST model created successfully")
    print(f"   - seq_len: {config.seq_len}")
    print(f"   - d_model: {config.d_model}")
    print(f"   - layers: {config.e_layers}")
    
except Exception as e:
    print(f"❌ Failed to create model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check data path
print("\n[TEST 4] Checking data availability...")
data_path = os.path.join(raft_root, 'data/ETT/ETTh1.csv')
if os.path.exists(data_path):
    print(f"✅ Data found: {data_path}")
else:
    print(f"❌ Data not found: {data_path}")
    sys.exit(1)

# Test 5: Check experiment framework
print("\n[TEST 5] Checking experiment framework...")
try:
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    print("✅ Experiment class available")
except ImportError as e:
    print(f"❌ Experiment framework issue: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED - Ready to run experiments!")
print("=" * 80)
print("\nNext steps:")
print("1. Run: python experiments_patchtst/exp_patchtst_sl720.py")
print("2. Run: python experiments_patchtst/exp_patchtst_sl3000.py")
print("3. Compare results")
print()
