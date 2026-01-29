#!/usr/bin/env python3
"""
Quick experiment runner to generate REAL loss curves for Graph 2
Runs only RAFT and PatchTST-720 with full epoch logging
"""

import os
import sys
import json
import time
import torch
import numpy as np
from datetime import datetime

# Add RAFT root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
raft_root = os.path.dirname(script_dir)
sys.path.insert(0, raft_root)

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def run_with_logging(config, experiment_name):
    """Run experiment and capture epoch-by-epoch losses"""
    
    print(f"\n{'='*80}")
    print(f"🚀 Running: {experiment_name}")
    print(f"{'='*80}\n")
    
    # Create experiment instance
    exp = Exp_Long_Term_Forecast(config)
    
    # Get data loaders
    train_data, train_loader = exp._get_data(flag='train')
    vali_data, vali_loader = exp._get_data(flag='val')
    
    # Storage for losses
    training_log = {
        'experiment_name': experiment_name,
        'model': config.model,
        'seq_len': config.seq_len,
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(f"📊 Training for {config.train_epochs} epochs...")
    start_time = time.time()
    
    # Setup optimizer and criterion
    model_optim = torch.optim.Adam(exp.model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    for epoch in range(config.train_epochs):
        epoch_start = time.time()
        
        # Train one epoch
        exp.model.train()
        train_losses = []
        for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()
            
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            
            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1).float().to(exp.device)
            
            # Forward pass
            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Compute loss
            f_dim = -1 if config.features == 'MS' else 0
            outputs = outputs[:, -config.pred_len:, f_dim:]
            batch_y = batch_y[:, -config.pred_len:, f_dim:].to(exp.device)
            loss = criterion(outputs, batch_y)
            train_losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            model_optim.step()
        
        train_loss = np.average(train_losses)
        
        # Validate
        exp.model.eval()
        vali_losses = []
        with torch.no_grad():
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(exp.device)
                batch_y = batch_y.float().to(exp.device)
                batch_x_mark = batch_x_mark.float().to(exp.device)
                batch_y_mark = batch_y_mark.float().to(exp.device)
                
                dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1).float().to(exp.device)
                
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if config.features == 'MS' else 0
                outputs = outputs[:, -config.pred_len:, f_dim:]
                batch_y = batch_y[:, -config.pred_len:, f_dim:].to(exp.device)
                loss = criterion(outputs, batch_y)
                vali_losses.append(loss.item())
        
        vali_loss = np.average(vali_losses)
        
        # Log
        training_log['epochs'].append(epoch + 1)
        training_log['train_losses'].append(float(train_loss))
        training_log['val_losses'].append(float(vali_loss))
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:2d}/{config.train_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {vali_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")
    
    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Training completed in {total_time:.2f} minutes")
    
    # Final test - create setting string for test method
    setting = f"{config.model_id}_{config.data}_{config.features}_{config.seq_len}_{config.pred_len}"
    test_loss, test_mae = exp.test(setting, test=0)
    
    training_log['final_test_mse'] = float(test_loss)
    training_log['final_test_mae'] = float(test_mae)
    training_log['training_time_minutes'] = total_time
    
    # Save logs
    os.makedirs('./results/training_logs', exist_ok=True)
    log_file = f"./results/training_logs/{experiment_name.replace(' ', '_').lower()}_losses.json"
    
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"📁 Saved training log: {log_file}")
    print(f"📊 Final Test MSE: {test_loss:.4f}, MAE: {test_mae:.4f}\n")
    
    return training_log


# ============================================================================
# Configuration 1: PatchTST-720
# ============================================================================

class PatchTST720Config:
    task_name = 'long_term_forecast'
    is_training = 1
    model_id = 'PatchTST_720_loss_curves'
    model = 'PatchTST'
    
    # Data
    data = 'ETTh1'
    root_path = '../data/ETT/'
    data_path = 'ETTh1.csv'
    features = 'M'
    target = 'OT'
    freq = 'h'
    checkpoints = '../checkpoints/'
    
    # Forecasting
    seq_len = 720
    label_len = 48
    pred_len = 96
    seasonal_patterns = 'Monthly'
    
    # PatchTST specific
    e_layers = 3
    d_layers = 1
    d_model = 128
    d_ff = 256
    n_heads = 16
    enc_in = 7
    dec_in = 7
    c_out = 7
    factor = 1
    dropout = 0.1
    embed = 'timeF'
    activation = 'gelu'
    output_attention = False
    
    # Patching parameters
    patch_len = 16
    stride = 8
    
    # Training
    num_workers = 0
    itr = 1
    train_epochs = 10
    batch_size = 32
    patience = 3
    learning_rate = 0.0001
    des = 'test'
    loss = 'MSE'
    lradj = 'type1'
    use_amp = False
    use_gpu = torch.cuda.is_available()
    gpu = 0
    use_multi_gpu = False
    devices = '0'
    
    # Augmentation
    augmentation_ratio = 0
    
    # Additional required attributes
    top_k = 5
    num_kernels = 6
    moving_avg = 25
    distil = True
    mix = True
    
    # RAFT-specific (even though this is PatchTST)
    num_chunks = 4
    chunk_len = 24
    time_cag_layers = 1
    forecast_decoder = False
    window_size = [4]
    
    # Inverse transform
    inverse = False
    
    # Model type flags
    do_predict = False
    cols = None


# ============================================================================
# Configuration 2: Vanilla Transformer 3000 (for worst-case comparison)
# ============================================================================

class Vanilla3000Config:
    task_name = 'long_term_forecast'
    is_training = 1
    model_id = 'PatchTST_3000_loss_curves'
    model = 'PatchTST'  # Use PatchTST instead of Transformer
    
    # Data
    data = 'ETTh1'
    root_path = '../data/ETT/'
    data_path = 'ETTh1.csv'
    features = 'M'
    target = 'OT'
    freq = 'h'
    checkpoints = '../checkpoints/'
    
    # Forecasting
    seq_len = 3000
    label_len = 48
    pred_len = 96
    seasonal_patterns = 'Monthly'
    
    # PatchTST specific
    e_layers = 3
    d_layers = 1
    d_model = 128
    d_ff = 256
    n_heads = 16
    enc_in = 7
    dec_in = 7
    c_out = 7
    factor = 1
    dropout = 0.1
    embed = 'timeF'
    activation = 'gelu'
    output_attention = False
    
    # Patching parameters
    patch_len = 16
    stride = 8
    
    # Training
    num_workers = 0
    itr = 1
    train_epochs = 10
    batch_size = 8  # Reduced from 32 for seq_len=3000 to avoid OOM
    patience = 3
    learning_rate = 0.0001
    des = 'test'
    loss = 'MSE'
    lradj = 'type1'
    use_amp = False
    use_gpu = torch.cuda.is_available()
    gpu = 0
    use_multi_gpu = False
    devices = '0'
    
    # Augmentation
    augmentation_ratio = 0
    
    # Additional required attributes
    top_k = 5
    num_kernels = 6
    moving_avg = 25
    distil = True
    mix = True
    
    # RAFT-specific (not used in Vanilla but required by framework)
    num_chunks = 4
    chunk_len = 24
    time_cag_layers = 1
    forecast_decoder = False
    window_size = [4]
    
    # Inverse transform
    inverse = False
    
    # Model type flags
    do_predict = False
    cols = None


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("📈 GENERATING REAL TRAINING LOSS CURVES FOR GRAPH 2")
    print("="*80)
    print("\nThis will run 2 experiments to capture real training curves:")
    print("  1. PatchTST-720 (SOTA baseline)")
    print("  2. PatchTST-3000 (long context degradation)")
    print("\nEstimated time: ~20-30 minutes total")
    print("="*80 + "\n")
    
    # Run experiments
    results = {}
    
    # Experiment 1: PatchTST-720
    results['patchtst_720'] = run_with_logging(
        PatchTST720Config(), 
        "PatchTST 720"
    )
    
    # Experiment 2: Vanilla-3000
    results['vanilla_3000'] = run_with_logging(
        Vanilla3000Config(),
        "PatchTST 3000"
    )
    
    print("\n" + "="*80)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\n📊 Summary:")
    for name, log in results.items():
        print(f"\n{log['experiment_name']}:")
        print(f"  Final Test MSE: {log['final_test_mse']:.4f}")
        print(f"  Training Time: {log['training_time_minutes']:.2f} min")
        print(f"  Epochs: {len(log['epochs'])}")
    
    print("\n📁 Training logs saved to: ./results/training_logs/")
    print("🎨 Now run: python generate_real_graph2.py")
    print("="*80 + "\n")
