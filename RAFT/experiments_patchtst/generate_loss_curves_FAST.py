#!/usr/bin/env python3
"""
FAST VERSION: Generate loss curves with just PatchTST
Uses only PatchTST-720 experiment (3 min) + RAFT baseline reference
"""

import os
import sys
import json
import time
import torch
from datetime import datetime

# Add RAFT root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
raft_root = os.path.dirname(script_dir)
sys.path.insert(0, raft_root)

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


class PatchTST720Config:
    task_name = 'long_term_forecast'
    is_training = 1
    model_id = 'PatchTST_720_loss_curves'
    model = 'PatchTST'
    
    data = 'ETTh1'
    root_path = './data/ETT/'
    data_path = 'ETTh1.csv'
    features = 'M'
    target = 'OT'
    freq = 'h'
    checkpoints = './checkpoints/'
    
    seq_len = 720
    label_len = 48
    pred_len = 96
    seasonal_patterns = 'Monthly'
    
    e_layers = 3
    d_layers = 1
    d_model = 128
    d_ff = 256
    n_heads = 16
    enc_in = 7
    dec_in = 7
    c_out = 7
    dropout = 0.1
    
    patch_len = 16
    stride = 8
    
    num_workers = 0
    train_epochs = 10
    batch_size = 32
    patience = 3
    learning_rate = 0.0001
    loss = 'MSE'
    use_gpu = torch.cuda.is_available()


def run_patchtst_with_logging():
    """Run PatchTST-720 and capture losses"""
    
    print("\n" + "="*80)
    print("🚀 Running PatchTST-720 with loss logging")
    print("="*80 + "\n")
    
    config = PatchTST720Config()
    exp = Exp_Long_Term_Forecast(config)
    
    train_loader = exp._get_data(flag='train')
    vali_loader = exp._get_data(flag='val')
    
    training_log = {
        'experiment_name': 'PatchTST 720',
        'model': 'PatchTST',
        'seq_len': 720,
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    start_time = time.time()
    
    for epoch in range(config.train_epochs):
        exp.model.train()
        train_loss = exp.train(train_loader)
        
        exp.model.eval()
        vali_loss = exp.vali(vali_loader, vali_loader)
        
        training_log['epochs'].append(epoch + 1)
        training_log['train_losses'].append(float(train_loss))
        training_log['val_losses'].append(float(vali_loss))
        
        print(f"Epoch {epoch+1:2d}/10 | Train: {train_loss:.4f} | Val: {vali_loss:.4f}")
    
    total_time = (time.time() - start_time) / 60
    
    test_loader = exp._get_data(flag='test')
    test_loss, test_mae = exp.test(test_loader)
    
    training_log['final_test_mse'] = float(test_loss)
    training_log['final_test_mae'] = float(test_mae)
    training_log['training_time_minutes'] = total_time
    
    os.makedirs('./results/training_logs', exist_ok=True)
    log_file = './results/training_logs/patchtst_720_losses.json'
    
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n✅ Done in {total_time:.2f} min | Test MSE: {test_loss:.4f}")
    print(f"📁 Saved: {log_file}\n")


if __name__ == '__main__':
    run_patchtst_with_logging()
    print("🎨 Now run: python generate_real_graph2_fast.py")
