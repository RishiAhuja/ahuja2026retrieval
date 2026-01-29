#!/usr/bin/env python3
"""
EXCHANGE DATASET VALIDATION - Cross-Domain Proof
=================================================
Financial time series validation with FIXED dataset loading.

Fix: Use args.data='custom' for Exchange dataset, not 'Exchange'
"""

import sys
import os
import time
import json
from datetime import datetime

# Add RAFT root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
raft_root = os.path.dirname(script_dir)
if raft_root not in sys.path:
    sys.path.insert(0, raft_root)

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_long_context_forecasting import Exp_LongContext_Forecast
import torch

def create_args(model, seq_len):
    """Create args for a specific model and sequence length"""
    class Args:
        def __init__(self):
            # FIXED: Exchange uses 'custom' dataset loader!
            self.data = 'custom'  # NOT 'Exchange'!
            self.root_path = './data/exchange_rate/'
            self.data_path = 'exchange_rate.csv'
            
            # Dataset configuration - Exchange specific
            self.features = 'M'
            self.target = 'OT'
            self.freq = 'd'
            self.enc_in = 8  # Exchange has 8 features
            self.dec_in = 8
            self.c_out = 8
            
            # Prediction setup
            self.seq_len = seq_len
            self.label_len = 48
            self.pred_len = 96
            self.seasonal_patterns = 'Monthly'
            self.inverse = False
            
            # Model
            self.model = model
            
            # Common parameters
            self.factor = 1
            self.output_attention = False
            self.patch_size = 12
            self.stride = 12
            
            # Training
            self.train_epochs = 10
            self.batch_size = 32
            self.patience = 3
            self.learning_rate = 0.0001
            self.lradj = 'type1'
            self.use_amp = False
            
            # Data loader
            self.num_workers = 4
            self.embed = 'timeF'
            
            # GPU
            self.use_gpu = True
            self.gpu = 0
            self.use_multi_gpu = False
            self.devices = '0'
            
            # Other
            self.augmentation_ratio = 0
            self.p_hidden_dims = [128, 128]
            self.p_hidden_layers = 2
            
            # Output
            self.checkpoints = './checkpoints/'
            self.des = f'Exchange_validation_{model}_{seq_len}'
            self.task_name = 'long_term_forecast'
            self.is_training = 1
            self.model_id = f'Exchange_{model}_{seq_len}'
            self.itr = 1
    
    return Args()

def configure_model_specific(args):
    """Add model-specific parameters"""
    if args.model == 'RAFT':
        # RAFT-specific
        args.d_model = 512
        args.n_heads = 8
        args.e_layers = 2
        args.d_layers = 1
        args.d_ff = 2048
        args.dropout = 0.1
        args.activation = 'gelu'
        
        # RAFT unique parameters
        args.n_period = 3  # Retrieval periods
        args.topm = 20     # Top-M retrieved windows
        
    elif args.model == 'PatchTST':
        # PatchTST-specific
        args.d_model = 128
        args.n_heads = 16
        args.e_layers = 3
        args.d_ff = 256
        args.dropout = 0.1
        args.patch_len = 16
        args.stride = 8
        args.activation = 'gelu'
        
    elif args.model == 'TransformerLongContext':
        # Time-CAG best configuration
        args.d_model = 128
        args.n_heads = 8
        args.e_layers = 3
        args.d_layers = 1
        args.d_ff = 512
        args.dropout = 0.1
        args.activation = 'gelu'
    
    return args

def run_experiment(model, seq_len):
    """Run a single experiment"""
    print(f"\n{'='*80}")
    print(f"💼 Exchange - {model} (seq_len={seq_len})")
    print(f"{'='*80}")
    
    args = create_args(model, seq_len)
    args = configure_model_specific(args)
    
    # Fix paths
    args.root_path = os.path.join(raft_root, args.root_path.lstrip('./'))
    args.checkpoints = os.path.join(raft_root, 'checkpoints/')
    
    start_time = time.time()
    
    try:
        # Select correct experiment class
        if model in ['RAFT', 'PatchTST']:
            exp = Exp_Long_Term_Forecast(args)
        else:
            exp = Exp_LongContext_Forecast(args)
        
        # Training
        setting = f'Exchange_validation_{model}_{seq_len}'
        print(f"\n📚 Training...")
        exp.train(setting)
        
        # Testing
        print(f"\n🧪 Testing...")
        mse, mae = exp.test(setting, test=1)
        
        end_time = time.time()
        training_time = (end_time - start_time) / 60
        
        result = {
            "dataset": "Exchange",
            "model": model,
            "seq_len": seq_len,
            "test_mse": float(mse),
            "test_mae": float(mae),
            "training_time_minutes": float(training_time)
        }
        
        print(f"\n✅ Complete - MSE: {mse:.6f}, MAE: {mae:.6f}, Time: {training_time:.2f} min")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "dataset": "Exchange",
            "model": model,
            "seq_len": seq_len,
            "error": str(e)
        }

def main():
    print("\n" + "=" * 80)
    print("💼 EXCHANGE DATASET VALIDATION - Cross-Domain Proof")
    print("=" * 80)
    print()
    print("🎯 Objective: Prove PatchTST degradation generalizes to financial data")
    print()
    print("🔧 Fix Applied: Using args.data='custom' (not 'Exchange')")
    print("   • Dataset: Exchange Rate (8 features, daily frequency)")
    print("   • 4 experiments: RAFT, PatchTST (720), PatchTST (3000), Time-CAG")
    print()
    print("⏱️  Expected total runtime: ~15-20 minutes")
    print("=" * 80)
    
    # Define experiments
    experiments = [
        ("RAFT", 720),
        ("PatchTST", 720),
        ("PatchTST", 3000),
        ("TransformerLongContext", 720),
    ]
    
    results = {}
    
    for i, (model, seq_len) in enumerate(experiments, 1):
        print(f"\n\n{'#'*80}")
        print(f"# EXPERIMENT {i}/4")
        print(f"{'#'*80}")
        
        exp_key = f"{model.lower()}_{seq_len}"
        results[exp_key] = run_experiment(model, seq_len)
        
        # Save results after each experiment
        os.makedirs('./experiments/results', exist_ok=True)
        result_file = './experiments/results/exchange_validation.json'
        with open(result_file, 'w') as f:
            json.dump({
                "dataset": "Exchange",
                "timestamp": datetime.now().isoformat(),
                "experiments": results
            }, f, indent=2)
        
        print(f"\n💾 Progress saved to: {result_file}")
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("📊 EXCHANGE VALIDATION SUMMARY")
    print("=" * 80)
    
    for exp_key, result in results.items():
        if 'error' not in result:
            print(f"\n{exp_key}:")
            print(f"  MSE: {result['test_mse']:.6f}")
            print(f"  MAE: {result['test_mae']:.6f}")
            print(f"  Time: {result['training_time_minutes']:.2f} min")
        else:
            print(f"\n{exp_key}: ❌ {result['error']}")
    
    # Calculate degradation
    if 'patchtst_720' in results and 'patchtst_3000' in results:
        if 'error' not in results['patchtst_720'] and 'error' not in results['patchtst_3000']:
            mse_720 = results['patchtst_720']['test_mse']
            mse_3000 = results['patchtst_3000']['test_mse']
            degradation = ((mse_3000 - mse_720) / mse_720) * 100
            
            print("\n" + "=" * 80)
            print("📈 PATCHTST DEGRADATION ANALYSIS (EXCHANGE)")
            print("=" * 80)
            print(f"  PatchTST (720):  MSE = {mse_720:.6f}")
            print(f"  PatchTST (3000): MSE = {mse_3000:.6f}")
            print(f"  Degradation: {degradation:+.1f}%")
            print()
            
            if degradation > 10:
                print("  ✅ CONFIRMED: Degradation generalizes to financial data!")
            elif degradation > 0:
                print("  ⚠️  Partial degradation observed")
            else:
                print("  ❌ No degradation on Exchange dataset")
            print("=" * 80)
    
    print(f"\n\n✅ All Exchange experiments complete!")
    print(f"📁 Results saved to: {result_file}\n")

if __name__ == '__main__':
    main()
