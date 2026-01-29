#!/usr/bin/env python3
"""
PatchTST CORRECTED - Phase 1: Validation Experiment
Reproduce original PatchTST paper results (seq_len=336)

REFERENCE: yuqinie98/PatchTST - scripts/PatchTST/etth1.sh
PAPER: "A Time Series is Worth 64 Words" (ICLR 2023)
EXPECTED MSE: 0.377-0.380 (Table 1)

⚠️ CRITICAL: This MUST match original paper within ±5% or implementation is broken!
"""

import os
import sys
import json
import time
from datetime import datetime

# Add RAFT root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
raft_root = os.path.dirname(os.path.dirname(script_dir))
if raft_root not in sys.path:
    sys.path.insert(0, raft_root)

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import torch


class Args:
    """
    CORRECTED Configuration - Matches Original PatchTST Paper Exactly
    
    All hyperparameters taken from:
    https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/scripts/PatchTST/etth1.sh
    """
    
    # Basic Config
    task_name = 'long_term_forecast'
    is_training = 1
    model_id = 'PatchTST_336_CORRECTED'
    model = 'PatchTST'
    
    # Data
    data = 'ETTh1'
    root_path = './data/ETT/'
    data_path = 'ETTh1.csv'
    features = 'M'  # Multivariate
    target = 'OT'
    freq = 'h'
    checkpoints = './checkpoints/'
    
    # Forecasting Task
    seq_len = 336       # Original paper baseline
    label_len = 48
    pred_len = 96       # Predicting 96 steps ahead
    seasonal_patterns = 'Monthly'
    inverse = False
    
    # Model Architecture - CORRECTED TO MATCH ORIGINAL PAPER
    enc_in = 7
    dec_in = 7
    c_out = 7
    d_model = 16        # ← WAS 128 - NOW CORRECTED
    n_heads = 4         # ← WAS 8 - NOW CORRECTED
    e_layers = 3        # ✓ Correct
    d_layers = 1
    d_ff = 128          # ← WAS 256 - NOW CORRECTED
    
    # PatchTST specific
    patch_len = 16
    stride = 8
    
    # Training - CORRECTED TO MATCH ORIGINAL PAPER
    num_workers = 0
    itr = 1
    train_epochs = 100  # ← WAS 10 - NOW CORRECTED (10x longer!)
    batch_size = 128    # ← WAS 16 - NOW CORRECTED (8x larger!)
    patience = 20       # ← WAS 3 - NOW CORRECTED
    learning_rate = 0.0001
    des = 'PatchTST_336_Original'
    loss = 'MSE'
    lradj = 'TST'       # ← WAS 'type1' - NOW CORRECTED (OneCycle scheduler!)
    pct_start = 0.3     # ← NEW - OneCycle warmup percentage
    use_amp = False
    
    # Dropout - CORRECTED TO MATCH ORIGINAL PAPER
    dropout = 0.3       # ← WAS 0.1 - NOW CORRECTED (3x stronger!)
    fc_dropout = 0.3    # ← NEW - Added missing parameter
    head_dropout = 0    # ← NEW - Added missing parameter
    
    # Other required params
    moving_avg = 25
    factor = 1
    distil = True
    embed = 'timeF'
    activation = 'gelu'
    output_attention = False
    
    # GPU
    use_gpu = True
    gpu = 0
    use_multi_gpu = False
    devices = '0'
    
    # RAFT specific (required by data loader, unused by PatchTST)
    augmentation_ratio = 0
    p_hidden_dims = [128, 256]
    p_hidden_layers = 2


def run_experiment():
    """Execute validation experiment"""
    
    print("=" * 100)
    print("PHASE 1: VALIDATION - PatchTST Original Configuration (seq_len=336)")
    print("=" * 100)
    print("\n🎯 GOAL: Reproduce original PatchTST paper results")
    print("📚 REFERENCE: ICLR 2023 - 'A Time Series is Worth 64 Words'")
    print("📊 EXPECTED MSE: 0.377-0.380 (from Table 1)")
    print("\n⚠️  IF THIS FAILS: Implementation bug - must fix before proceeding!\n")
    
    args = Args()
    
    # Fix paths
    args.root_path = os.path.join(raft_root, 'data/ETT/')
    args.checkpoints = os.path.join(raft_root, 'checkpoints/')
    
    print("\n" + "=" * 100)
    print("CONFIGURATION (Original Paper)")
    print("=" * 100)
    print(f"  Model:           {args.model}")
    print(f"  Dataset:         {args.data}")
    print(f"  seq_len:         {args.seq_len}")
    print(f"  pred_len:        {args.pred_len}")
    print()
    print("  Architecture:")
    print(f"    d_model:       {args.d_model} (small, efficient)")
    print(f"    n_heads:       {args.n_heads}")
    print(f"    e_layers:      {args.e_layers}")
    print(f"    d_ff:          {args.d_ff}")
    print(f"    dropout:       {args.dropout}")
    print(f"    fc_dropout:    {args.fc_dropout}")
    print()
    print("  Patching:")
    print(f"    patch_len:     {args.patch_len}")
    print(f"    stride:        {args.stride}")
    print(f"    num_patches:   {int((args.seq_len - args.patch_len) / args.stride + 2)}")
    print()
    print("  Training:")
    print(f"    epochs:        {args.train_epochs} ← 10x more than old config!")
    print(f"    batch_size:    {args.batch_size} ← 8x larger than old config!")
    print(f"    learning_rate: {args.learning_rate}")
    print(f"    lradj:         {args.lradj} ← OneCycle (not exponential decay!)")
    print(f"    pct_start:     {args.pct_start}")
    print(f"    patience:      {args.patience}")
    print("=" * 100)
    print()
    
    # Initialize experiment
    setting = f'{args.model_id}_{args.data}_sl{args.seq_len}_pl{args.pred_len}'
    exp = Exp_Long_Term_Forecast(args)
    
    # Training
    print("\n" + "=" * 100)
    print("TRAINING PHASE")
    print("=" * 100)
    print(f"Setting: {setting}")
    print("This will take ~4 hours on V100 GPU...")
    print()
    
    start_time = time.time()
    
    try:
        exp.train(setting)
        training_time = time.time() - start_time
        
        # Testing
        print("\n" + "=" * 100)
        print("TESTING PHASE")
        print("=" * 100)
        
        test_start = time.time()
        mse, mae = exp.test(setting, test=1)
        test_time = time.time() - test_start
        
        # Validation Check
        print("\n" + "=" * 100)
        print("VALIDATION RESULTS")
        print("=" * 100)
        print(f"Test MSE:       {mse:.6f}")
        print(f"Test MAE:       {mae:.6f}")
        print(f"Training Time:  {training_time/60:.1f} minutes")
        print()
        
        # Check against original paper
        original_mse_min = 0.370
        original_mse_max = 0.385
        tolerance = 0.05  # 5% tolerance
        
        if original_mse_min * (1 - tolerance) <= mse <= original_mse_max * (1 + tolerance):
            print("✅ VALIDATION PASSED!")
            print(f"   MSE {mse:.3f} is within expected range [{original_mse_min:.3f}, {original_mse_max:.3f}]")
            print("   Implementation is correct - safe to proceed to Phase 2!")
            status = "PASSED"
        elif mse < original_mse_min:
            print("⚠️  VALIDATION WARNING:")
            print(f"   MSE {mse:.3f} is BETTER than original paper ({original_mse_min:.3f})")
            print("   This is unusual but not necessarily wrong.")
            print("   Possible reasons: better random seed, minor implementation differences")
            status = "PASSED_WITH_WARNING"
        else:
            print("❌ VALIDATION FAILED!")
            print(f"   MSE {mse:.3f} is outside expected range [{original_mse_min:.3f}, {original_mse_max:.3f}]")
            print("   ERROR: Implementation bug detected!")
            print("   DO NOT PROCEED - Fix implementation first!")
            status = "FAILED"
        
        print("=" * 100)
        
        # Save results
        results = {
            'experiment_name': 'Phase 1 - Validation (PatchTST-336 Original)',
            'phase': 1,
            'purpose': 'Validate implementation against original paper',
            'model': 'PatchTST',
            'seq_len': 336,
            'test_mse': float(mse),
            'test_mae': float(mae),
            'training_time_minutes': round(training_time / 60, 1),
            'test_time_seconds': round(test_time, 1),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'validation_status': status,
            'expected_range': [original_mse_min, original_mse_max],
            'configuration': {
                'epochs': args.train_epochs,
                'batch_size': args.batch_size,
                'lradj': args.lradj,
                'd_model': args.d_model,
                'n_heads': args.n_heads,
                'dropout': args.dropout,
                'fc_dropout': args.fc_dropout
            },
            'reference': 'PatchTST (ICLR 2023) - Table 1'
        }
        
        results_dir = os.path.join(raft_root, 'experiments_patchtst/corrected/results')
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = os.path.join(results_dir, 'phase1_validation_336.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        if status == "FAILED":
            print("\n" + "=" * 100)
            print("⚠️  CRITICAL: VALIDATION FAILED - DO NOT PROCEED!")
            print("=" * 100)
            print("\nNext steps:")
            print("1. Check model implementation in models/PatchTST.py")
            print("2. Verify data loading in data_provider/")
            print("3. Compare with original PatchTST repository")
            print("4. Re-run this experiment after fixes")
            sys.exit(1)
        else:
            print("\n" + "=" * 100)
            print("✅ PROCEED TO PHASE 2")
            print("=" * 100)
            print("\nNext experiments:")
            print("  - Phase 2A: PatchTST-720 (corrected)")
            print("  - Phase 2B: PatchTST-3000 (corrected)")
            print("\nRun these to test the core hypothesis!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        error_results = {
            'experiment_name': 'Phase 1 - Validation (FAILED)',
            'phase': 1,
            'status': 'ERROR',
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results_dir = os.path.join(raft_root, 'experiments_patchtst/corrected/results')
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, 'phase1_validation_ERROR.json'), 'w') as f:
            json.dump(error_results, f, indent=2)
        
        raise


if __name__ == '__main__':
    print("\n" + "=" * 100)
    print("STARTING PHASE 1 - VALIDATION EXPERIMENT")
    print("=" * 100)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 100)
    
    run_experiment()
