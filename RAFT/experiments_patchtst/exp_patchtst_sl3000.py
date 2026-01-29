#!/usr/bin/env python3
"""
⚠️  DEPRECATED - DO NOT USE THIS FILE!

This configuration has CRITICAL ERRORS:
- train_epochs = 10 (should be 100)
- lradj = 'type1' (should be 'TST')
- batch_size = 8 (should be 128)
- d_model = 128 (should be 16)

USE INSTEAD: experiments_patchtst/corrected/phase2b_patchtst_3000_corrected_CRITICAL.py

See CRITICAL_HYPERPARAMETER_AUDIT.md for details.

RESULTS FROM THIS FILE ARE INVALID due to severe undertraining and wrong LR schedule!

=== OLD (INCORRECT) CODE BELOW ===

PatchTST Experiment 2: Long Context (seq_len=3000)
CRITICAL TEST: Does PatchTST degrade with longer context?
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

# Add RAFT root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
raft_root = os.path.dirname(script_dir)
if raft_root not in sys.path:
    sys.path.insert(0, raft_root)

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import torch

# Import training logger
try:
    from utils.training_logger import TrainingLogger
except ImportError:
    print("ERROR: Could not import TrainingLogger")
    sys.exit(1)


class Args:
    """Configuration for PatchTST at seq_len=3000"""
    
    # Basic Config
    task_name = 'long_term_forecast'
    is_training = 1
    model_id = 'PatchTST_3000'
    model = 'PatchTST'
    
    # Data
    data = 'ETTh1'
    root_path = './data/ETT/'
    data_path = 'ETTh1.csv'
    features = 'M'
    target = 'OT'
    freq = 'h'
    checkpoints = './checkpoints/'
    
    # Forecasting Task
    seq_len = 3000
    label_len = 48
    pred_len = 96
    seasonal_patterns = 'Monthly'
    inverse = False
    
    # Model Architecture
    enc_in = 7
    dec_in = 7
    c_out = 7
    d_model = 128
    n_heads = 8
    e_layers = 3
    d_layers = 1
    d_ff = 256
    
    # PatchTST specific
    patch_len = 16
    stride = 8
    
    # Training (smaller batch for memory)
    num_workers = 0
    itr = 1
    train_epochs = 10
    batch_size = 8  # Reduced for 3000 seq_len
    patience = 3
    learning_rate = 0.0001
    des = 'PatchTST_3000'
    loss = 'MSE'
    lradj = 'type1'
    use_amp = False
    dropout = 0.1
    
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
    
    # RAFT specific (required by data loader)
    augmentation_ratio = 0
    p_hidden_dims = [128, 256]
    p_hidden_layers = 2


def run_experiment():
    """Execute PatchTST experiment at seq_len=3000"""
    
    print("=" * 80)
    print("PATCHTST EXPERIMENT 2: Long Context (seq_len=3000)")
    print("=" * 80)
    print("\n⚠️  CRITICAL TEST: Does PatchTST degrade with longer context?")
    print()
    
    args = Args()
    
    # Fix paths
    args.root_path = os.path.join(raft_root, 'data/ETT/')
    args.checkpoints = os.path.join(raft_root, 'checkpoints/')
    
    # Initialize logger
    logger = TrainingLogger(log_dir='./experiments_patchtst/results/training_logs')
    logger.start_experiment('PatchTST (3000)')
    
    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  seq_len: {args.seq_len} (4.2× longer than RAFT)")
    print(f"  patch_len: {args.patch_len}")
    print(f"  stride: {args.stride}")
    print(f"  d_model: {args.d_model}")
    print(f"  e_layers: {args.e_layers}")
    print(f"  train_epochs: {args.train_epochs}")
    print(f"  batch_size: {args.batch_size} (reduced for memory)")
    print()
    
    # Initialize experiment
    exp = Exp_Long_Term_Forecast(args)
    
    # Training
    print("\n[TRAINING PHASE]")
    start_time = time.time()
    
    setting = f'{args.model_id}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}'
    
    try:
        print(f'Training: {setting}')
        exp.train(setting)
        
        training_time = time.time() - start_time
        
        # Testing
        print("\n[TESTING PHASE]")
        test_start = time.time()
        mse, mae = exp.test(setting, test=1)
        test_time = time.time() - test_start
        
        # Results
        print("\n" + "=" * 80)
        print("RESULTS - PatchTST (3000)")
        print("=" * 80)
        print(f"Test MSE:       {mse:.6f}")
        print(f"Test MAE:       {mae:.6f}")
        print(f"Training Time:  {training_time/60:.2f} minutes")
        print("=" * 80)
        print("\nFull Comparison:")
        print(f"  RAFT (720):                 0.379 MSE ⭐")
        print(f"  Vanilla Transformer (720):  0.556 MSE")
        print(f"  Vanilla Transformer (3000): 1.323 MSE ❌ (degraded)")
        print(f"  PatchTST (3000):            {mse:.3f} MSE", end="")
        
        if mse < 0.5:
            print(" ✅ (STABLE - patching works!)")
            conclusion = "PatchTST handles long context well - patching mechanism effective"
        elif mse < 0.8:
            print(" ⚠️  (MILD DEGRADATION)")
            conclusion = "PatchTST shows some degradation but better than vanilla"
        else:
            print(" ❌ (SEVERE DEGRADATION)")
            conclusion = "Even PatchTST degrades with long context - MAJOR FINDING!"
        
        print("=" * 80)
        print(f"\n🔬 Analysis: {conclusion}")
        print("=" * 80)
        
        # Save results
        results = {
            'experiment_name': 'PatchTST (3000)',
            'model': 'PatchTST',
            'seq_len': 3000,
            'test_mse': float(mse),
            'test_mae': float(mae),
            'training_time_minutes': round(training_time / 60, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'comparison': {
                'RAFT_720': 0.379,
                'Vanilla_720': 0.556,
                'Vanilla_3000': 1.323,
                'PatchTST_3000': float(mse)
            },
            'conclusion': conclusion,
            'degradation_vs_raft': f"{((mse - 0.379) / 0.379 * 100):.1f}%"
        }
        
        results_dir = './experiments_patchtst/results'
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, 'patchtst_3000_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to experiments_patchtst/results/")
        
        # Decision guidance
        print("\n" + "=" * 80)
        print("📋 NEXT STEPS:")
        print("=" * 80)
        if mse > 0.8:
            print("✅ Run exp_patchtst_sl1440.py to track degradation trend")
            print("✅ Write paper emphasizing: 'Even SOTA fails at long context'")
        elif mse > 0.5:
            print("⚠️  Run exp_patchtst_sl1440.py to understand partial degradation")
            print("⚠️  Frame as: 'Retrieval + patching may be complementary'")
        else:
            print("📊 PatchTST handles long context - reframe paper:")
            print("   - Retrieval offers efficiency (4× less memory)")
            print("   - Slightly better accuracy")
            print("   - Consider hybrid retrieval + patching")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    run_experiment()
