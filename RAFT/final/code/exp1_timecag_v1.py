#!/usr/bin/env python3
"""
Experiment 1: Time-CAG v1 (Baseline Configuration)
===================================================

Configuration:
- Model: TransformerLongContext
- seq_len: 3000 (full history)
- d_model: 256
- e_layers: 3
- dropout: 0.1
- patch_size: 12
- batch_size: 8
- train_epochs: 10

Expected Outcome: Establish baseline Time-CAG performance
"""

import os
import sys
import json
import time
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

class Args:
    """Configuration for Experiment 1"""
    
    # Basic Config
    task_name = 'long_term_forecast'
    is_training = 1
    model_id = 'ETTh1_exp1'
    model = 'TransformerLongContext'
    
    # Data
    data = 'ETTh1'
    root_path = './data/ETT/'
    data_path = 'ETTh1.csv'
    features = 'M'  # Multivariate
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
    enc_in = 7  # ETTh1 has 7 features
    dec_in = 7
    c_out = 7
    d_model = 256
    n_heads = 8
    e_layers = 3
    d_layers = 1
    d_ff = 512
    moving_avg = 25
    factor = 1
    distil = True
    dropout = 0.1
    embed = 'timeF'
    activation = 'gelu'
    output_attention = False
    
    # Patching (PatchTST style)
    patch_size = 12
    stride = 12
    
    # RAFT specific (not used in long context)
    augmentation_ratio = 0
    p_hidden_dims = [128, 256]
    p_hidden_layers = 2
    
    # Training
    num_workers = 0
    itr = 1
    train_epochs = 10
    batch_size = 8
    patience = 3
    learning_rate = 0.0001
    des = 'Exp1_TimeCAG_v1'
    loss = 'MSE'
    lradj = 'type1'
    use_amp = False
    
    # GPU
    use_gpu = True
    gpu = 0
    use_multi_gpu = False
    devices = '0'
    
    # De-stationary Projector Params
    p_hidden_dims = [128, 256]
    p_hidden_layers = 2


def run_experiment():
    """Execute Experiment 1 and log results"""
    
    print("=" * 80)
    print("EXPERIMENT 1: Time-CAG v1 (Baseline Configuration)")
    print("=" * 80)
    
    args = Args()
    
    # Fix data path to use RAFT root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raft_root = os.path.dirname(script_dir)
    args.root_path = os.path.join(raft_root, 'data/ETT/')
    args.checkpoints = os.path.join(raft_root, 'checkpoints/')
    
    # Log configuration
    config = {
        'experiment_id': 1,
        'experiment_name': 'Time-CAG v1',
        'model': args.model,
        'seq_len': args.seq_len,
        'd_model': args.d_model,
        'e_layers': args.e_layers,
        'dropout': args.dropout,
        'patch_size': args.patch_size,
        'batch_size': args.batch_size,
        'train_epochs': args.train_epochs,
        'learning_rate': args.learning_rate,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize experiment
    exp = Exp_LongContext_Forecast(args)
    
    # Training
    print("\n[TRAINING PHASE]")
    start_time = time.time()
    
    try:
        train_losses = []
        val_losses = []
        
        # Train the model (this will run for train_epochs)
        setting = f'{args.model_id}_{args.model}_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_el{args.e_layers}'
        
        print(f'Training: {setting}')
        exp.train(setting)
        
        training_time = time.time() - start_time
        
        # Testing
        print("\n[TESTING PHASE]")
        test_start = time.time()
        
        mse, mae = exp.test(setting, test=1)
        
        test_time = time.time() - test_start
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            'experiment_id': 1,
            'experiment_name': 'Time-CAG v1',
            'status': 'SUCCESS',
            'configuration': config,
            'results': {
                'test_mse': float(mse),
                'test_mae': float(mae),
                'training_time_minutes': round(training_time / 60, 2),
                'test_time_minutes': round(test_time / 60, 2),
                'total_time_minutes': round(total_time / 60, 2)
            },
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print results
        print("\n" + "=" * 80)
        print("RESULTS - EXPERIMENT 1")
        print("=" * 80)
        print(f"Test MSE:       {mse:.6f}")
        print(f"Test MAE:       {mae:.6f}")
        print(f"Training Time:  {training_time/60:.2f} minutes")
        print(f"Total Time:     {total_time/60:.2f} minutes")
        print("=" * 80)
        
        # Save results to JSON
        results_dir = './experiments/results'
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, 'exp1_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_file}")
        
        # Append to master log
        master_log = os.path.join(results_dir, 'all_experiments.jsonl')
        with open(master_log, 'a') as f:
            f.write(json.dumps(results) + '\n')
        
        print(f"✅ Appended to master log: {master_log}")
        
        return results
        
    except Exception as e:
        error_results = {
            'experiment_id': 1,
            'experiment_name': 'Time-CAG v1',
            'status': 'FAILED',
            'error': str(e),
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print("\n" + "=" * 80)
        print("❌ EXPERIMENT FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        
        # Save error log
        results_dir = './experiments/results'
        os.makedirs(results_dir, exist_ok=True)
        error_file = os.path.join(results_dir, 'exp1_error.json')
        with open(error_file, 'w') as f:
            json.dump(error_results, f, indent=2)
        
        raise


if __name__ == '__main__':
    run_experiment()
