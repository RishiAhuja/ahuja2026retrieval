#!/usr/bin/env python3
"""
Experiment 2: Time-CAG v2 (Medium Context)
==========================================
Configuration:
- seq_len: 1440 (2x RAFT context)
- pred_len: 96
- d_model: 128
- e_layers: 3
- dropout: 0.1
- train_epochs: 10
- batch_size: 16

Goal: Test medium context length (2x RAFT)
Expected: Better than v1, but may still struggle
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

from data_provider.data_factory import data_provider
from exp.exp_long_context_forecasting import Exp_LongContext_Forecast
import torch
import numpy as np

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
        
        # Experiment 2 specific settings
        self.seq_len = 1440  # Medium context (2x RAFT)
        self.label_len = 48
        self.pred_len = 96
        self.seasonal_patterns = 'Monthly'
        self.inverse = False
        self.task_name = 'long_term_forecast' 

        
        # Model parameters
        self.d_model = 128
        self.n_heads = 8
        self.e_layers = 3
        self.d_ff = 512
        self.dropout = 0.1
        self.activation = 'gelu'
        
        # Patching
        self.patch_size = 12
        self.stride = 12
        
        # Training
        self.train_epochs = 10
        self.batch_size = 16  # Larger than exp1 (shorter context)
        self.patience = 3
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
        self.des = 'Exp2_TimeCag_v2'
        self.task_name = 'long_term_forecast'
        self.is_training = 1
        self.model_id = 'ETTh1_exp2'
        self.itr = 1


def run_experiment():
    print("=" * 80)
    print("EXPERIMENT 2: Time-CAG v2 (Medium Context = 1440)")
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
    elif args.use_Context_Forecastds.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Mac GPU)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    exp = Exp_Long_Context_Forecasting(args)
    
    print(f"\nModel: {args.model}")
    print(f"Context Length: {args.seq_len} (2x RAFT)")
    print(f"Prediction Length: {args.pred_len}")
    print(f"Batch Size: {args.batch_size}")
    print()
    
    print("Starting training...")
    print("-" * 80)
    
    setting = f'ETTh1_exp2_{args.model}_{args.data}_M_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_el{args.e_layers}'
    exp.train(setting)
    
    print("\nRunning final test...")
    print("-" * 80)
    mse, mae = exp.test(setting, test=1)
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    
    results = {
        "experiment": "Experiment 2: Time-CAG v2",
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
        }
    }
    
    os.makedirs('./experiments/results', exist_ok=True)
    result_file = './experiments/results/exp2_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 80)
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Training Time: {training_time:.2f} minutes")
    print(f"Results saved to: {result_file}")
    print("=" * 80)
    
    return results

if __name__ == '__main__':
    run_experiment()
