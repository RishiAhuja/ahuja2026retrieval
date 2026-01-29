#!/usr/bin/env python3
"""
PatchTST Experiment 1: Baseline (seq_len=720)
Fair comparison against RAFT at same context length
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
    """Configuration for PatchTST at seq_len=720"""
    
    # Basic Config
    task_name = 'long_term_forecast'
    is_training = 1
    model_id = 'PatchTST_720'
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
    seq_len = 720
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
    
    # Training
    num_workers = 0
    itr = 1
    train_epochs = 10
    batch_size = 16
    patience = 3
    learning_rate = 0.0001
    des = 'PatchTST_720'
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
    """Execute PatchTST experiment at seq_len=720"""
    
    print("=" * 80)
    print("PATCHTST EXPERIMENT 1: Baseline (seq_len=720)")
    print("=" * 80)
    
    args = Args()
    
    # Fix paths
    args.root_path = os.path.join(raft_root, 'data/ETT/')
    args.checkpoints = os.path.join(raft_root, 'checkpoints/')
    
    # Initialize logger
    logger = TrainingLogger(log_dir='./experiments_patchtst/results/training_logs')
    logger.start_experiment('PatchTST (720)')
    
    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  seq_len: {args.seq_len}")
    print(f"  patch_len: {args.patch_len}")
    print(f"  stride: {args.stride}")
    print(f"  d_model: {args.d_model}")
    print(f"  e_layers: {args.e_layers}")
    print(f"  train_epochs: {args.train_epochs}")
    print(f"  Comparison: RAFT (720) = 0.379 MSE")
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
        print("RESULTS - PatchTST (720)")
        print("=" * 80)
        print(f"Test MSE:       {mse:.6f}")
        print(f"Test MAE:       {mae:.6f}")
        print(f"Training Time:  {training_time/60:.2f} minutes")
        print("=" * 80)
        print("\nComparison:")
        print(f"  RAFT (720):              0.379 MSE")
        print(f"  PatchTST (720):          {mse:.3f} MSE")
        print(f"  Vanilla Transformer (720): 0.556 MSE")
        print("=" * 80)
        
        # Save results
        results = {
            'experiment_name': 'PatchTST (720)',
            'model': 'PatchTST',
            'seq_len': 720,
            'test_mse': float(mse),
            'test_mae': float(mae),
            'training_time_minutes': round(training_time / 60, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'comparison': {
                'RAFT_720': 0.379,
                'PatchTST_720': float(mse),
                'Vanilla_720': 0.556
            }
        }
        
        results_dir = './experiments_patchtst/results'
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, 'patchtst_720_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to experiments_patchtst/results/")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    run_experiment()
