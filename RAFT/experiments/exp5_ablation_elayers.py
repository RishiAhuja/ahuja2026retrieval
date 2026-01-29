#!/usr/bin/env python3
"""
Experiment 5: Ablation Study - Encoder Layers (e_layers)
=========================================================
Tests: e_layers = [1, 2, 3, 4]
Fixed: seq_len=720, d_model=128, dropout=0.1

Goal: Find optimal model depth
Expected: Deeper may overfit, shallower may underfit
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

def create_args(e_layers):
    class Args:
        def __init__(self):
            self.model = 'TransformerLongContext'
            self.data = 'ETTh1'
            self.root_path = './data/ETT/'
            self.data_path = 'ETTh1.csv'
            self.features = 'M'
            self.target = 'OT'
            self.freq = 'h'
            
            # Fixed settings
            self.seq_len = 720
            self.label_len = 48
            self.pred_len = 96
            self.seasonal_patterns = 'Monthly'
            self.inverse = False
            
            # Variable: e_layers
            self.d_model = 128
            self.n_heads = 8
            self.e_layers = e_layers
            self.d_ff = 512
            self.dropout = 0.1
            self.activation = 'gelu'
            
            self.patch_size = 12
            self.stride = 12
            
            self.train_epochs = 10
            self.batch_size = 32
            self.patience = 3
            self.learning_rate = 0.0001
            self.lradj = 'type1'
            self.use_amp = False
            
            self.num_workers = 0
            self.embed = 'timeF'
            self.enc_in = 7
            self.dec_in = 7
            self.c_out = 7
            
            self.use_gpu = True
            self.gpu = 0
            self.use_multi_gpu = False
            self.devices = '0'
            
            self.augmentation_ratio = 0
            self.top_k = 5
            self.d_state = 16
            self.d_conv = 4
            self.expand = 2
            self.p_hidden_dims = [128, 128]
            self.p_hidden_layers = 2
            
            self.checkpoints = './checkpoints/'
            self.des = f'Exp5_Ablation_elayers_{e_layers}'
            self.task_name = 'long_term_forecast'
            self.is_training = 1
            self.model_id = f'ETTh1_exp5_el{e_layers}'
            self.itr = 1
    
    return Args()

def run_experiment():
    print("=" * 80)
    print("EXPERIMENT 5: Ablation Study - Encoder Layers (e_layers)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    e_layers_values = [1, 2, 3, 4]
    all_results = {}
    
    for e_layers in e_layers_values:
        print("\n" + "=" * 80)
        print(f"Testing e_layers = {e_layers}")
        print("=" * 80)
        
        args = create_args(e_layers)
        
        # Fix data path to use RAFT root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        raft_root = os.path.dirname(script_dir)
        args.root_path = os.path.join(raft_root, 'data/ETT/')
        args.checkpoints = os.path.join(raft_root, 'checkpoints/')
        
        start_time = time.time()
        
        # Set device
        if args.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda:0')
        elif args.use_gpu and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        exp = Exp_LongContext_Forecast(args)
        
        print(f"Encoder Layers: {e_layers}")
        print(f"d_model: {args.d_model}")
        print()
        
        print("Training...")
        setting = f'ETTh1_exp5_el{e_layers}_{args.model}_{args.data}_M_sl{args.seq_len}_pl{args.pred_len}'
        exp.train(setting)
        
        print("\nTesting...")
        mse, mae = exp.test(setting, test=1)
        
        end_time = time.time()
        training_time = (end_time - start_time) / 60
        
        all_results[f'e_layers_{e_layers}'] = {
            "e_layers": e_layers,
            "test_mse": float(mse),
            "test_mae": float(mae),
            "training_time_minutes": float(training_time),
        }
        
        print(f"✓ e_layers={e_layers}: MSE={mse:.6f}, MAE={mae:.6f}, Time={training_time:.2f}min")
    
    # Save all results
    results = {
        "experiment": "Experiment 5: Ablation - e_layers",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "fixed_settings": {
            "seq_len": 720,
            "d_model": 128,
            "dropout": 0.1,
        },
        "results": all_results
    }
    
    os.makedirs('./experiments/results', exist_ok=True)
    result_file = './experiments/results/exp5_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 5 COMPLETE - SUMMARY")
    print("=" * 80)
    for key, val in all_results.items():
        print(f"e_layers={val['e_layers']}: MSE={val['test_mse']:.6f}, Time={val['training_time_minutes']:.2f}min")
    print(f"\nResults saved to: {result_file}")
    print("=" * 80)
    
    return results

if __name__ == '__main__':
    run_experiment()
