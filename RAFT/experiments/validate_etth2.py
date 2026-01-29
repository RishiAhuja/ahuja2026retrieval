#!/usr/bin/env python3
"""
CROSS-DOMAIN VALIDATION: ETTh2 + Exchange Rate
================================================
Run core experiments on 2 validation datasets to prove generalization

Datasets:
- ETTh2: Electricity (different transformer than ETTh1)  
- Exchange: Financial (8 currency exchange rates - different domain)

Experiments per dataset (4 runs):
1. RAFT (720) - Winner baseline
2. PatchTST (720) - SOTA at short context
3. PatchTST (3000) - Degradation proof (MAIN RESULT)
4. Time-CAG best (720) - Transferred hyperparameters

Expected runtime: ~1.5 hours per dataset, ~3 hours total (8 experiments)
Goal: Prove long-context degradation generalizes beyond ETTh1
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

from exp.exp_long_context_forecasting import Exp_LongContext_Forecast
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import torch

def create_args(model, seq_len, dataset='ETTh2', d_model=None, e_layers=None):
    """Create args for different model configurations"""
    class Args:
        def __init__(self):
            # Dataset configuration
            self.data = dataset
            
            if dataset == 'ETTh2':
                self.root_path = './data/ETT/'
                self.data_path = 'ETTh2.csv'
                self.features = 'M'
                self.target = 'OT'
                self.freq = 'h'
                self.enc_in = 7
                self.dec_in = 7
                self.c_out = 7
            elif dataset == 'Exchange':
                self.root_path = './data/exchange_rate/'
                self.data_path = 'exchange_rate.csv'
                self.features = 'M'
                self.target = 'OT'
                self.freq = 'h'  # Business day frequency
                self.enc_in = 8  # 8 exchange rate features
                self.dec_in = 8
                self.c_out = 8
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
            
            # Prediction setup (same as ETTh1 for comparison)
            self.seq_len = seq_len
            self.label_len = 48
            self.pred_len = 96
            self.seasonal_patterns = 'Monthly'
            self.inverse = False
            
            # Model
            self.model = model
            
            # Architecture (set based on model type)
            if model == 'RAFT':
                self.d_model = 512
                self.n_heads = 8
                self.e_layers = 2
                self.d_ff = 2048
                self.dropout = 0.1
                self.top_k = 5
                self.d_state = 16
                self.d_conv = 4
                self.expand = 2
                # RAFT specific parameters
                self.n_period = 3  # Number of periods for retrieval
                self.topm = 20     # Number of top retrievals
            elif model == 'TransformerLongContext':
                self.d_model = d_model or 32  # Use best config from ETTh1
                self.n_heads = 8 if self.d_model >= 64 else 4
                self.e_layers = e_layers or 2
                self.d_ff = self.d_model * 4
                self.dropout = 0.1
            elif model == 'PatchTST':
                self.d_model = 128
                self.n_heads = 16
                self.e_layers = 3
                self.d_ff = 256
                self.dropout = 0.1
                self.patch_len = 16
                self.stride = 8
            else:  # Fallback
                self.d_model = 512
                self.n_heads = 8
                self.e_layers = 2
                self.d_ff = 2048
                self.dropout = 0.1
            
            self.activation = 'gelu'
            
            # Common parameters for all models
            self.factor = 1  # Attention factor
            self.output_attention = False
            self.patch_size = 12  # Used by some models
            self.stride = 12      # Used by some models
            
            # Training
            self.train_epochs = 10
            self.batch_size = 32
            self.patience = 3
            self.learning_rate = 0.0001
            self.lradj = 'type1'
            self.use_amp = False
            
            # Data loader
            self.num_workers = 0
            self.embed = 'timeF'
            # enc_in, dec_in, c_out set above based on dataset
            
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
            self.des = f'{dataset}_validation_{model}_sl{seq_len}'
            self.task_name = 'long_term_forecast'
            self.is_training = 1
            self.model_id = f'{dataset}_{model}_sl{seq_len}'
            self.itr = 1
    
    return Args()

def run_experiment(exp_name, model, seq_len, dataset='ETTh2', d_model=None, e_layers=None):
    """Run a single experiment and return results"""
    print("\n" + "=" * 80)
    print(f"🧪 {exp_name}")
    print(f"Dataset: {dataset} | Model: {model} | Context: {seq_len} tokens")
    print("=" * 80)
    
    args = create_args(model, seq_len, dataset, d_model, e_layers)
    
    # Fix paths
    args.root_path = os.path.join(raft_root, args.root_path.lstrip('./'))  
    args.checkpoints = os.path.join(raft_root, 'checkpoints/')
    
    start_time = time.time()
    
    # Select the correct experiment class based on model
    # RAFT and PatchTST use Exp_Long_Term_Forecast (different forward signature)
    # TransformerLongContext uses Exp_LongContext_Forecast
    try:
        if model in ['RAFT', 'PatchTST']:
            exp = Exp_Long_Term_Forecast(args)
        else:
            exp = Exp_LongContext_Forecast(args)
    except Exception as e:
        print(f"⚠️  Model initialization failed: {e}")
        print(f"   Model name: {model}")
        print(f"   args.model: {args.model}")
        import traceback
        traceback.print_exc()
        raise
    
    # Train
    print("\n📚 Training...")
    setting = f'{dataset}_validation_{model}_sl{seq_len}'
    exp.train(setting)
    
    # Test
    print("\n🧪 Testing...")
    mse, mae = exp.test(setting, test=1)
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    
    result = {
        "experiment": exp_name,
        "dataset": dataset,
        "model": model,
        "seq_len": seq_len,
        "d_model": d_model or args.d_model,
        "e_layers": e_layers or args.e_layers,
        "test_mse": float(mse),
        "test_mae": float(mae),
        "training_time_minutes": float(training_time)
    }
    
    print(f"\n✅ {exp_name} COMPLETE")
    print(f"   MSE: {mse:.6f} | MAE: {mae:.6f} | Time: {training_time:.2f}min")
    
    return result

def run_dataset_experiments(dataset_name):
    """Run all core experiments for a single dataset"""
    print("\n" + "=" * 80)
    print(f"📊 DATASET: {dataset_name}")
    print("=" * 80)
    
    results = {}
    
    # Experiment 1: RAFT Baseline (720)
    try:
        result = run_experiment(
            f"{dataset_name} - Exp1: RAFT Baseline (720)", 
            model="RAFT", 
            seq_len=720,
            dataset=dataset_name
        )
        results['raft_720'] = result
    except Exception as e:
        print(f"❌ RAFT-720 failed: {e}")
        results['raft_720'] = {"error": str(e)}
    
    # Experiment 2: PatchTST (720) - SOTA at short context
    try:
        result = run_experiment(
            f"{dataset_name} - Exp2: PatchTST (720)", 
            model="PatchTST", 
            seq_len=720,
            dataset=dataset_name
        )
        results['patchtst_720'] = result
    except Exception as e:
        print(f"❌ PatchTST-720 failed: {e}")
        results['patchtst_720'] = {"error": str(e)}
    
    # Experiment 3: PatchTST (3000) - DEGRADATION PROOF ⭐
    try:
        result = run_experiment(
            f"{dataset_name} - Exp3: PatchTST (3000)", 
            model="PatchTST", 
            seq_len=3000,
            dataset=dataset_name
        )
        results['patchtst_3000'] = result
    except Exception as e:
        print(f"❌ PatchTST-3000 failed: {e}")
        results['patchtst_3000'] = {"error": str(e)}
    
    # Experiment 4: Time-CAG Best Config (720) - Transferred hyperparameters
    try:
        result = run_experiment(
            f"{dataset_name} - Exp4: Time-CAG Best (720)", 
            model="TransformerLongContext", 
            seq_len=720,
            dataset=dataset_name,
            d_model=32,
            e_layers=2
        )
        results['timecag_best'] = result
    except Exception as e:
        print(f"❌ Time-CAG failed: {e}")
        results['timecag_best'] = {"error": str(e)}
    
    return results

def main():
    """Main function to run cross-domain validation"""
    print("\n" + "=" * 80)
    print("🔬 CROSS-DOMAIN VALIDATION: ETTh2 + Exchange Rate")
    print("=" * 80)
    print()
    print("📦 Datasets:")
    print("  1. ETTh2 (Electricity - different transformer than ETTh1)")
    print("  2. Exchange (Finance - 8 currency rates, different domain)")
    print()
    print("🧪 Experiments per dataset:")
    print("  1. RAFT (720) - Winner baseline")
    print("  2. PatchTST (720) - SOTA at short context")
    print("  3. PatchTST (3000) - Degradation proof ⭐")
    print("  4. Time-CAG best (720) - Transferred hyperparameters")
    print()
    print("Expected runtime: ~1.5 hours per dataset, ~3 hours total (8 experiments)")
    print("=" * 80)
    
    all_results = {}
    
    # Run ETTh2 experiments
    print("\n\n" + "🔵" * 40)
    print("PHASE 1: ETTh2 (Electricity Domain)")
    print("🔵" * 40)
    etth2_results = run_dataset_experiments('ETTh2')
    all_results['ETTh2'] = etth2_results
    
    # Run Exchange experiments  
    print("\n\n" + "🟢" * 40)
    print("PHASE 2: Exchange Rate (Financial Domain)")
    print("🟢" * 40)
    exchange_results = run_dataset_experiments('Exchange')
    all_results['Exchange'] = exchange_results
    
    # Save comprehensive results
    results_summary = {
        "purpose": "Cross-domain validation to verify findings generalize",
        "datasets": ["ETTh2", "Exchange"],
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "results_by_dataset": all_results
    }
    
    os.makedirs('./experiments/results', exist_ok=True)
    result_file = './experiments/results/cross_domain_validation.json'
    with open(result_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print comprehensive comparison
    print("\n\n" + "=" * 80)
    print("📊 FINAL RESULTS: Cross-Domain Validation")
    print("=" * 80)
    
    for dataset_name, results in all_results.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")
        print(f"{'Model':<30} {'Seq Len':<12} {'MSE':<12} {'Time (min)':<12}")
        print("-" * 80)
        
        for exp_key, result in results.items():
            if "error" not in result:
                print(f"{result['model']:<30} {result['seq_len']:<12} "
                      f"{result['test_mse']:<12.6f} {result['training_time_minutes']:<12.2f}")
        
        # Calculate degradation if both PatchTST runs succeeded
        if ('patchtst_720' in results and 'patchtst_3000' in results and
            "error" not in results['patchtst_720'] and "error" not in results['patchtst_3000']):
            mse_720 = results['patchtst_720']['test_mse']
            mse_3000 = results['patchtst_3000']['test_mse']
            degradation_pct = ((mse_3000 - mse_720) / mse_720) * 100
            print()
            print(f"⚠️  PatchTST Degradation (720→3000): {degradation_pct:+.1f}%")
            if degradation_pct > 10:
                print(f"   ✅ CONFIRMED: Long context degrades on {dataset_name}")
            else:
                print(f"   ⚠️  WARNING: Minimal degradation on {dataset_name}")
    
    # Cross-dataset summary
    print("\n" + "=" * 80)
    print("🎯 CROSS-DATASET SUMMARY")
    print("=" * 80)
    print("Key Findings:")
    
    datasets_with_degradation = 0
    for dataset_name, results in all_results.items():
        if ('patchtst_720' in results and 'patchtst_3000' in results and
            "error" not in results['patchtst_720'] and "error" not in results['patchtst_3000']):
            mse_720 = results['patchtst_720']['test_mse']
            mse_3000 = results['patchtst_3000']['test_mse']
            degradation_pct = ((mse_3000 - mse_720) / mse_720) * 100
            print(f"  • {dataset_name}: {degradation_pct:+.1f}% degradation (720→3000)")
            if degradation_pct > 10:
                datasets_with_degradation += 1
    
    print()
    if datasets_with_degradation >= 2:
        print("✅ SUCCESS: Long-context degradation generalizes across domains!")
        print("   → Paper claim validated on multiple datasets")
    elif datasets_with_degradation == 1:
        print("⚠️  PARTIAL: Degradation found in some datasets only")
        print("   → May be domain-specific phenomenon")
    else:
        print("❌ FAILURE: No consistent degradation pattern")
        print("   → Hypothesis may not generalize")
    
    print()
    print(f"📁 Results saved to: {result_file}")
    print("=" * 80)

if __name__ == '__main__':
    main()
