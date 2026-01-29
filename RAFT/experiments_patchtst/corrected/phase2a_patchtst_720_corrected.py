#!/usr/bin/env python3
"""
PatchTST CORRECTED - Phase 2A: Fair Comparison at seq_len=720

CRITICAL TEST: How does PatchTST perform at seq_len=720 with CORRECT configuration?

REFERENCE: Original config from yuqinie98/PatchTST
COMPARISON: Against RAFT-720 (MSE=0.379)
"""

import os
import sys
import json
import time
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
raft_root = os.path.dirname(os.path.dirname(script_dir))
if raft_root not in sys.path:
    sys.path.insert(0, raft_root)

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import torch


class Args:
    """CORRECTED Configuration for seq_len=720"""
    
    # Basic Config
    task_name = 'long_term_forecast'
    is_training = 1
    model_id = 'PatchTST_720_CORRECTED'
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
    seq_len = 720       # RAFT comparison point
    label_len = 48
    pred_len = 96
    seasonal_patterns = 'Monthly'
    inverse = False
    
    # Model Architecture - CORRECTED
    enc_in = 7
    dec_in = 7
    c_out = 7
    d_model = 16        # ← CORRECTED (was 128)
    n_heads = 4         # ← CORRECTED (was 8)
    e_layers = 3
    d_layers = 1
    d_ff = 128          # ← CORRECTED (was 256)
    
    # PatchTST specific
    patch_len = 16
    stride = 8
    
    # Training - CORRECTED
    num_workers = 0
    itr = 1
    train_epochs = 100  # ← CORRECTED (was 10)
    batch_size = 128    # ← CORRECTED (was 16)
    patience = 20       # ← CORRECTED (was 3)
    learning_rate = 0.0001
    des = 'PatchTST_720_Corrected'
    loss = 'MSE'
    lradj = 'TST'       # ← CORRECTED (was 'type1')
    pct_start = 0.3
    use_amp = False
    
    # Dropout - CORRECTED
    dropout = 0.3       # ← CORRECTED (was 0.1)
    fc_dropout = 0.3    # ← NEW
    head_dropout = 0    # ← NEW
    
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
    
    # RAFT specific
    augmentation_ratio = 0
    p_hidden_dims = [128, 256]
    p_hidden_layers = 2


def run_experiment():
    """Execute Phase 2A experiment"""
    
    print("=" * 100)
    print("PHASE 2A: PatchTST at seq_len=720 (CORRECTED Configuration)")
    print("=" * 100)
    print("\n🎯 GOAL: Fair comparison against RAFT at same context length")
    print("📊 BASELINE: RAFT-720 MSE = 0.379")
    print("\n⚠️  This uses 100 epochs + OneCycle LR (not 10 epochs + type1!)\n")
    
    args = Args()
    args.root_path = os.path.join(raft_root, 'data/ETT/')
    args.checkpoints = os.path.join(raft_root, 'checkpoints/')
    
    print("\n" + "=" * 100)
    print("CONFIGURATION CHANGES")
    print("=" * 100)
    print("  Parameter         OLD (Wrong)    NEW (Correct)   Change")
    print("  " + "-" * 96)
    print(f"  train_epochs      10             {args.train_epochs}             ← 10x more!")
    print(f"  batch_size        16             {args.batch_size}            ← 8x larger!")
    print(f"  lradj             'type1'        '{args.lradj}'           ← OneCycle!")
    print(f"  d_model           128            {args.d_model}             ← 8x smaller!")
    print(f"  n_heads           8              {args.n_heads}              ← 2x fewer!")
    print(f"  dropout           0.1            {args.dropout}            ← 3x stronger!")
    print(f"  fc_dropout        (missing)      {args.fc_dropout}            ← Added!")
    print("=" * 100)
    print()
    
    print(f"Expected patches: {int((args.seq_len - args.patch_len) / args.stride + 2)}")
    print(f"Expected runtime: ~3-4 hours on V100")
    print()
    
    setting = f'{args.model_id}_{args.data}_sl{args.seq_len}_pl{args.pred_len}'
    exp = Exp_Long_Term_Forecast(args)
    
    print("\n" + "=" * 100)
    print("TRAINING PHASE")
    print("=" * 100)
    
    start_time = time.time()
    
    try:
        exp.train(setting)
        training_time = time.time() - start_time
        
        print("\n" + "=" * 100)
        print("TESTING PHASE")
        print("=" * 100)
        
        test_start = time.time()
        mse, mae = exp.test(setting, test=1)
        test_time = time.time() - test_start
        
        print("\n" + "=" * 100)
        print("PHASE 2A RESULTS")
        print("=" * 100)
        print(f"Test MSE:       {mse:.6f}")
        print(f"Test MAE:       {mae:.6f}")
        print(f"Training Time:  {training_time/60:.1f} minutes")
        print()
        print("Comparison:")
        print(f"  RAFT-720:         0.379 MSE")
        print(f"  PatchTST-720:     {mse:.3f} MSE", end="")
        
        if mse < 0.379:
            print(" ✅ (Better than RAFT!)")
            status = "BETTER_THAN_RAFT"
        elif mse < 0.420:
            print(" ⚠️  (Slightly worse than RAFT)")
            status = "COMPARABLE_TO_RAFT"
        else:
            print(" ❌ (Significantly worse than RAFT)")
            status = "WORSE_THAN_RAFT"
        
        print("=" * 100)
        
        # Save results
        results = {
            'experiment_name': 'Phase 2A - PatchTST-720 (Corrected)',
            'phase': '2A',
            'purpose': 'Fair comparison at seq_len=720',
            'model': 'PatchTST',
            'seq_len': 720,
            'test_mse': float(mse),
            'test_mae': float(mae),
            'training_time_minutes': round(training_time / 60, 1),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': status,
            'comparison': {
                'RAFT_720': 0.379,
                'PatchTST_720_corrected': float(mse),
                'difference': float(mse - 0.379),
                'percent_diff': float((mse - 0.379) / 0.379 * 100)
            },
            'configuration': {
                'epochs': args.train_epochs,
                'batch_size': args.batch_size,
                'lradj': args.lradj,
                'd_model': args.d_model,
                'dropout': args.dropout
            }
        }
        
        results_dir = os.path.join(raft_root, 'experiments_patchtst/corrected/results')
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = os.path.join(results_dir, 'phase2a_patchtst_720_corrected.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        print("\n✅ PROCEED TO PHASE 2B (seq_len=3000)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    run_experiment()
