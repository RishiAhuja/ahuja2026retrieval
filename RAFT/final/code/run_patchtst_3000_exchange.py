#!/usr/bin/env python3
"""
MEMORY-OPTIMIZED: PatchTST 3000 on Exchange Dataset
====================================================
Run ONLY the PatchTST 3000 experiment with aggressive memory cleanup.
This completes the Exchange cross-domain validation.

Run this on a clean server restart with no other experiments running.
"""

import sys
import os
import time
import json
from datetime import datetime
import gc

# Add RAFT root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
raft_root = os.path.dirname(script_dir)
if raft_root not in sys.path:
    sys.path.insert(0, raft_root)

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import torch

def create_args():
    """Create args for PatchTST 3000 on Exchange"""
    class Args:
        def __init__(self):
            # CRITICAL: Exchange uses 'custom' dataset loader!
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
            self.seq_len = 3000  # LONG CONTEXT - degradation test
            self.label_len = 48
            self.pred_len = 96
            self.seasonal_patterns = 'Monthly'
            self.inverse = False
            
            # Model
            self.model = 'PatchTST'
            
            # PatchTST Architecture - REDUCED for memory
            self.d_model = 128
            self.n_heads = 16
            self.e_layers = 3
            self.d_ff = 256
            self.dropout = 0.1
            self.patch_len = 16
            self.stride = 8
            self.activation = 'gelu'
            
            # Common parameters
            self.factor = 1
            self.output_attention = False
            self.patch_size = 12
            self.stride = 12
            
            # Training - HEAVILY REDUCED batch size for memory
            self.train_epochs = 10
            self.batch_size = 8  # Reduced from 32 to save memory (Exchange has less data)
            self.patience = 3
            self.learning_rate = 0.0001
            self.lradj = 'type1'
            self.use_amp = False
            
            # Data loader
            self.num_workers = 0  # Reduce memory overhead
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
            self.des = 'Exchange_PatchTST_3000_memopt'
            self.task_name = 'long_term_forecast'
            self.is_training = 1
            self.model_id = 'Exchange_PatchTST_3000'
            self.itr = 1
    
    return Args()

def main():
    print("\n" + "=" * 80)
    print("💼 MEMORY-OPTIMIZED: PatchTST 3000 on Exchange Dataset")
    print("=" * 80)
    print("\n🎯 This completes the cross-domain validation!")
    print("   Comparing with PatchTST 720 MSE: 0.0928")
    print()
    print("💾 Memory optimizations:")
    print("   • Batch size reduced: 32 → 8 (Exchange has smaller dataset)")
    print("   • Aggressive garbage collection")
    print("   • Manual CUDA cache clearing")
    print()
    print("⏱️  Expected runtime: ~3-5 minutes")
    print("=" * 80)
    
    # Clear any existing GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\n🧹 GPU Memory cleared")
        print(f"   Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    args = create_args()
    
    # Fix paths
    args.root_path = os.path.join(raft_root, args.root_path.lstrip('./'))  
    args.checkpoints = os.path.join(raft_root, 'checkpoints/')
    
    start_time = time.time()
    
    print("\n🔧 Initializing PatchTST model...")
    exp = Exp_Long_Term_Forecast(args)
    
    # Training
    print("\n📚 Training PatchTST with 3000 context length on Exchange...")
    print("   (Testing if long context degrades on financial data)")
    setting = 'Exchange_PatchTST_3000_memopt'
    
    try:
        exp.train(setting)
        
        # Clear memory before testing
        torch.cuda.empty_cache()
        gc.collect()
        
        # Testing
        print("\n🧪 Testing...")
        mse, mae = exp.test(setting, test=1)
        
        end_time = time.time()
        training_time = (end_time - start_time) / 60
        
        # Calculate degradation vs PatchTST 720
        mse_720 = 0.0928  # From Exchange results
        degradation_pct = ((mse - mse_720) / mse_720) * 100
        
        result = {
            "experiment": "Exchange - PatchTST (3000) - Degradation Proof",
            "dataset": "Exchange",
            "model": "PatchTST",
            "seq_len": 3000,
            "d_model": 128,
            "e_layers": 3,
            "batch_size": 8,
            "test_mse": float(mse),
            "test_mae": float(mae),
            "training_time_minutes": float(training_time),
            "comparison": {
                "patchtst_720_mse": mse_720,
                "degradation_pct": float(degradation_pct)
            }
        }
        
        # Load existing results and update
        result_file = './experiments/results/exchange_validation.json'
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                all_results = json.load(f)
            
            # Update the patchtst_3000 entry with success
            all_results['experiments']['patchtst_3000'] = {
                "dataset": "Exchange",
                "model": "PatchTST",
                "seq_len": 3000,
                "test_mse": float(mse),
                "test_mae": float(mae),
                "training_time_minutes": float(training_time)
            }
            all_results['timestamp'] = datetime.now().isoformat()
        else:
            all_results = {
                "dataset": "Exchange",
                "timestamp": datetime.now().isoformat(),
                "experiments": {
                    "patchtst_3000": {
                        "dataset": "Exchange",
                        "model": "PatchTST",
                        "seq_len": 3000,
                        "test_mse": float(mse),
                        "test_mae": float(mae),
                        "training_time_minutes": float(training_time)
                    }
                }
            }
        
        # Save updated results
        with open(result_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Also save standalone result
        standalone_file = './experiments/results/patchtst_3000_exchange.json'
        with open(standalone_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Print results
        print("\n" + "=" * 80)
        print("✅ EXPERIMENT COMPLETE")
        print("=" * 80)
        print(f"\n📊 Results:")
        print(f"   Test MSE: {mse:.6f}")
        print(f"   Test MAE: {mae:.6f}")
        print(f"   Training Time: {training_time:.2f} minutes")
        print()
        print(f"📈 Degradation Analysis (EXCHANGE):")
        print(f"   PatchTST (720):  MSE = {mse_720:.6f}")
        print(f"   PatchTST (3000): MSE = {mse:.6f}")
        print(f"   Degradation: {degradation_pct:+.1f}%")
        print()
        
        if degradation_pct > 10:
            print("   ✅ CONFIRMED: Long context degrades on financial data!")
            print("      → Hypothesis validated across domains (ETT + Finance)")
        elif degradation_pct > 0:
            print("   ⚠️  PARTIAL: Some degradation observed")
            print("      → Effect present but weaker than expected")
        else:
            print("   ❌ NO DEGRADATION: Long context helps or neutral")
            print("      → Hypothesis not confirmed on Exchange")
        
        print()
        print(f"📁 Results updated in: {result_file}")
        print(f"📁 Standalone result: {standalone_file}")
        print("=" * 80)
        
        # Print full Exchange summary if we have all results
        if 'raft_720' in all_results['experiments']:
            print("\n" + "=" * 80)
            print("💼 COMPLETE EXCHANGE VALIDATION SUMMARY")
            print("=" * 80)
            for exp_key, exp_result in all_results['experiments'].items():
                if 'error' not in exp_result:
                    print(f"\n{exp_key}:")
                    print(f"  MSE: {exp_result['test_mse']:.6f}")
                    print(f"  MAE: {exp_result['test_mae']:.6f}")
            print("\n" + "=" * 80)
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ CUDA OUT OF MEMORY!")
            print("\n💡 Try these solutions:")
            print("   1. Restart server to clear all GPU memory")
            print("   2. Reduce batch_size further (currently 8, try 4)")
            print("   3. Reduce d_model to 64 or e_layers to 2")
            print("   4. Reduce seq_len to 2000 (still tests degradation)")
            print(f"\nError: {e}")
        else:
            raise

if __name__ == '__main__':
    main()
