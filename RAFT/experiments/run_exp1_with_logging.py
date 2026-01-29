#!/usr/bin/env python3
"""
Experiment 1 WITH REAL TRAINING LOGGING
This version captures actual training/validation losses
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
from exp.exp_long_context_forecasting import Exp_LongContext_Forecast
from utils.tools import EarlyStopping, adjust_learning_rate
import torch

# Import training logger (path added dynamically above)
try:
    from utils.training_logger import TrainingLogger
except ImportError:
    print("ERROR: Could not import TrainingLogger. Make sure utils/training_logger.py exists")
    sys.exit(1)


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
    
    # Patching
    patch_size = 12
    stride = 12
    
    # RAFT specific (not used in long context, but required by data loader)
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


def train_with_logging(exp, setting, logger):
    """
    Modified training loop that captures losses
    """
    train_data, train_loader = exp._get_data(flag='train')
    vali_data, vali_loader = exp._get_data(flag='val')
    test_data, test_loader = exp._get_data(flag='test')
    
    path = os.path.join(exp.args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)
    
    time_now = time.time()
    
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=exp.args.patience, verbose=True)
    
    model_optim = exp._select_optimizer()
    criterion = exp._select_criterion()
    
    for epoch in range(exp.args.train_epochs):
        iter_count = 0
        train_loss_list = []
        
        exp.model.train()
        epoch_time = time.time()
        
        # Training
        for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            
            # Forward - TransformerLongContext uses encoder only
            outputs = exp.model(batch_x, batch_x_mark)
            
            # Get predictions for forecast horizon
            f_dim = -1 if exp.args.features == 'MS' else 0
            outputs = outputs[:, -exp.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -exp.args.pred_len:, f_dim:]
            
            loss = criterion(outputs, batch_y)
            train_loss_list.append(loss.item())
            
            # Backward
            loss.backward()
            model_optim.step()
            
            if (i + 1) % 100 == 0:
                print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                speed = (time.time() - time_now) / iter_count
                print(f"\tspeed: {speed:.4f}s/iter")
        
        # Calculate average training loss
        train_loss = np.average(train_loss_list)
        
        # Validation
        vali_loss = validate(exp, vali_loader, criterion)
        
        # Log losses
        logger.log_epoch(epoch + 1, train_loss, vali_loss)
        
        print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.7f} | Vali Loss: {vali_loss:.7f}")
        
        # Early stopping check
        early_stopping(vali_loss, exp.model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        # Learning rate adjustment
        adjust_learning_rate(model_optim, epoch + 1, exp.args)
    
    # Load best model
    best_model_path = path + '/' + 'checkpoint.pth'
    exp.model.load_state_dict(torch.load(best_model_path))
    
    return exp.model


def validate(exp, vali_loader, criterion):
    """Validation function"""
    exp.model.eval()
    total_loss = []
    
    with torch.no_grad():
        for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            
            # Forward - TransformerLongContext uses encoder only
            outputs = exp.model(batch_x, batch_x_mark)
            
            # Get predictions for forecast horizon
            f_dim = -1 if exp.args.features == 'MS' else 0
            outputs = outputs[:, -exp.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -exp.args.pred_len:, f_dim:]
            
            loss = criterion(outputs, batch_y)
            total_loss.append(loss.item())
    
    vali_loss = np.average(total_loss)
    exp.model.train()
    return vali_loss


def run_experiment():
    """Execute Experiment 1 with real logging"""
    
    print("=" * 80)
    print("EXPERIMENT 1: Time-CAG v1 WITH REAL TRAINING LOGGING")
    print("=" * 80)
    
    args = Args()
    
    # Fix paths
    args.root_path = os.path.join(raft_root, 'data/ETT/')
    args.checkpoints = os.path.join(raft_root, 'checkpoints/')
    
    # Initialize logger
    logger = TrainingLogger()
    logger.start_experiment('Time-CAG v1 (3000)')
    
    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  seq_len: {args.seq_len}")
    print(f"  d_model: {args.d_model}")
    print(f"  e_layers: {args.e_layers}")
    print(f"  train_epochs: {args.train_epochs}")
    print()
    
    # Initialize experiment
    exp = Exp_LongContext_Forecast(args)
    
    # Training with logging
    print("\n[TRAINING PHASE WITH LOGGING]")
    start_time = time.time()
    
    setting = f'{args.model_id}_{args.model}_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_el{args.e_layers}'
    
    try:
        print(f'Training: {setting}')
        train_with_logging(exp, setting, logger)
        
        training_time = time.time() - start_time
        
        # Save training logs
        logger.save()
        
        # Testing
        print("\n[TESTING PHASE]")
        test_start = time.time()
        mse, mae = exp.test(setting, test=1)
        test_time = time.time() - test_start
        
        # Results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Test MSE:       {mse:.6f}")
        print(f"Test MAE:       {mae:.6f}")
        print(f"Training Time:  {training_time/60:.2f} minutes")
        print("=" * 80)
        
        # Save results
        results = {
            'experiment_name': 'Time-CAG v1 (3000)',
            'test_mse': float(mse),
            'test_mae': float(mae),
            'training_time_minutes': round(training_time / 60, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results_dir = './experiments/results'
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, 'exp1_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved!")
        print(f"✅ Training logs saved with REAL losses!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    run_experiment()
