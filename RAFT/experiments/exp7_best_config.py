#!/usr/bin/env python3
"""
Experiment 7: Best Configuration (Based on Ablation Results)
============================================================
Configuration:
- seq_len: 720 (fair comparison with RAFT)
- d_model: 64 (sweet spot - not too large)
- e_layers: 2 (moderate depth)
- dropout: 0.2 (stronger regularization)
- train_epochs: 15 (more training)
- batch_size: 32

Goal: Use best settings from ablation studies
Expected: Best possible Time-CAG performance
"""

import sys
import os
import time
import json
from datetime import datetime

# Add RAFT root to path (works in Colab and local)
script_dir = os.path.dirname(os.path.abspath(__file__))
raft_root = os.path.dirname(script_dir)
if raft_root not in sys.path:
    sys.path.insert(0, raft_root)
print(f"RAFT root: {raft_root}")

from exp.exp_long_context_forecasting import Exp_LongContext_Forecast
import torch

class Args:
    def __init__(self):
        # Basic config
        self.model = 'TransformerLongContext'
        self.data = 'ETTh1'
        self.root_path = './data/ETT/'
        self.data_path = 'ETTh1.csv'
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'
        
        # Best configuration (update after ablations)
        self.seq_len = 720  # Same as RAFT
        self.label_len = 48
        self.pred_len = 96
        self.seasonal_patterns = 'Monthly'
        self.inverse = False
        
        # Optimized hyperparameters
        self.d_model = 64  # Moderate capacity
        self.n_heads = 8
        self.e_layers = 2  # Not too deep
        self.d_ff = 256
        self.dropout = 0.2  # Stronger regularization
        self.activation = 'gelu'
        
        # Patching
        self.patch_size = 12
        self.stride = 12
        
        # Training (longer)
        self.train_epochs = 15
        self.batch_size = 32
        self.patience = 5
        self.learning_rate = 0.0001
        self.lradj = 'type1'
        self.use_amp = False
        
        # Data loader
        self.num_workers = 0
        self.embed = 'timeF'
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        
        # GPU
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0'
        
        # RAFT specific (not used)
        self.augmentation_ratio = 0
        self.top_k = 5
        self.d_state = 16
        self.d_conv = 4
        self.expand = 2
        self.p_hidden_dims = [128, 128]
        self.p_hidden_layers = 2
        
        # Output
        self.checkpoints = './checkpoints/'
        self.des = 'Exp7_BestConfig'
        self.task_name = 'long_term_forecast'
        self.is_training = 1
        self.model_id = 'ETTh1_exp7'
        self.itr = 1

def run_experiment():
    print("=" * 80)
    print("EXPERIMENT 7: Best Configuration (Optimized Time-CAG)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    args = Args()
    
    # Fix data path to use RAFT root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raft_root = os.path.dirname(script_dir)
    args.root_path = os.path.join(raft_root, 'data/ETT/')
    args.checkpoints = os.path.join(raft_root, 'checkpoints/')
    
    start_time = time.time()
    
    # Set device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Using CUDA")
    elif args.use_gpu and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Mac GPU)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    exp = Exp_LongContext_Forecast(args)
    
    print(f"\n🎯 OPTIMIZED CONFIGURATION:")
    print(f"Context Length: {args.seq_len} (same as RAFT)")
    print(f"Model Dimension: {args.d_model} (reduced capacity)")
    print(f"Encoder Layers: {args.e_layers} (moderate depth)")
    print(f"Dropout: {args.dropout} (stronger regularization)")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.train_epochs} (extended training)")
    print()
    print("🎯 Target: Beat RAFT's MSE of 0.379")
    print()
    
    print("Starting training...")
    print("-" * 80)
    
    setting = f'ETTh1_exp7_{args.model}_{args.data}_M_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_el{args.e_layers}'
    exp.train(setting)
    
    print("\nRunning final test...")
    print("-" * 80)
    mse, mae = exp.test(setting, test=1)
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    
    # Calculate performance vs RAFT
    raft_mse = 0.379
    improvement = ((raft_mse - mse) / raft_mse) * 100
    
    results = {
        "experiment": "Experiment 7: Best Configuration",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "configuration": {
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "d_model": args.d_model,
            "e_layers": args.e_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "train_epochs": args.train_epochs,
        },
        "results": {
            "test_mse": float(mse),
            "test_mae": float(mae),
            "training_time_minutes": float(training_time),
            "raft_baseline_mse": raft_mse,
            "improvement_vs_raft_percent": float(improvement),
            "better_than_raft": bool(mse < raft_mse),
        }
    }
    
    os.makedirs('./experiments/results', exist_ok=True)
    result_file = './experiments/results/exp7_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 7 COMPLETE - FINAL RESULTS")
    print("=" * 80)
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Training Time: {training_time:.2f} minutes")
    print()
    print(f"RAFT Baseline: {raft_mse:.6f}")
    if mse < raft_mse:
        print(f"✅ SUCCESS! Improved by {improvement:.2f}%")
    else:
        print(f"❌ Failed. Worse by {-improvement:.2f}%")
    print(f"\nResults saved to: {result_file}")
    print("=" * 80)
    
    return results

if __name__ == '__main__':
    run_experiment()
