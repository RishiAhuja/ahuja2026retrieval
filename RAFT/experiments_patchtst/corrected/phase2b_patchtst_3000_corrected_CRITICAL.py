#!/usr/bin/env python3
"""
PatchTST CORRECTED - Phase 2B: Critical Test at seq_len=3000

🚨 CRITICAL TEST: Does PatchTST degrade with longer context (with CORRECT config)?

This is THE experiment that determines our paper's validity:
- Scenario A: MSE < 0.45 → Original paper correct, our claim wrong
- Scenario B: MSE > 0.50 → Our finding validated, novel contribution!

REFERENCE: Original PatchTST should improve with longer context
OUR CLAIM: PatchTST degrades 68% when moving from 720 to 3000
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
    """CORRECTED Configuration for seq_len=3000"""
    
    # Basic Config
    task_name = 'long_term_forecast'
    is_training = 1
    model_id = 'PatchTST_3000_CORRECTED'
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
    seq_len = 3000      # 4.2x longer than RAFT baseline
    label_len = 48
    pred_len = 96
    seasonal_patterns = 'Monthly'
    inverse = False
    
    # Model Architecture - CORRECTED (SAME as 720 for fair comparison!)
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
    
    # Training - CORRECTED (SAME as 720 for fair comparison!)
    num_workers = 0
    itr = 1
    train_epochs = 100  # ← CORRECTED (was 10)
    batch_size = 128    # ← CORRECTED (was 8!) - NO REDUCTION!
    patience = 20       # ← CORRECTED (was 3)
    learning_rate = 0.0001
    des = 'PatchTST_3000_Corrected'
    loss = 'MSE'
    lradj = 'TST'       # ← CORRECTED (was 'type1')
    pct_start = 0.3
    use_amp = False
    
    # Dropout - CORRECTED (SAME as 720!)
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
    """Execute Phase 2B experiment - THE CRITICAL TEST"""
    
    print("=" * 100)
    print("🚨 PHASE 2B: CRITICAL TEST - PatchTST at seq_len=3000 (CORRECTED)")
    print("=" * 100)
    print("\n🎯 GOAL: Determine if PatchTST truly degrades with longer context")
    print("\n📊 TWO POSSIBLE OUTCOMES:")
    print()
    print("  Scenario A: MSE < 0.45")
    print("    → Original paper correct (patching helps at long context)")
    print("    → Our original finding was due to config errors")
    print("    → Paper needs MAJOR revision")
    print()
    print("  Scenario B: MSE > 0.50")
    print("    → Our finding VALIDATED (patching fails at extreme context)")
    print("    → Novel contribution confirmed")
    print("    → Paper is valid!")
    print()
    print("=" * 100)
    print()
    
    args = Args()
    args.root_path = os.path.join(raft_root, 'data/ETT/')
    args.checkpoints = os.path.join(raft_root, 'checkpoints/')
    
    print("\n" + "=" * 100)
    print("CONFIGURATION CHANGES FROM OLD (WRONG) VERSION")
    print("=" * 100)
    print("  Parameter         OLD (Wrong)    NEW (Correct)   Impact")
    print("  " + "-" * 96)
    print(f"  train_epochs      10             {args.train_epochs}             ← 10x more training!")
    print(f"  batch_size        8              {args.batch_size}            ← 16x larger! (was halved)")
    print(f"  lradj             'type1'        '{args.lradj}'           ← OneCycle (not decay!)")
    print(f"  d_model           128            {args.d_model}             ← 8x smaller model!")
    print(f"  n_heads           8              {args.n_heads}              ← 2x fewer heads!")
    print(f"  d_ff              256            {args.d_ff}            ← 2x smaller FFN!")
    print(f"  dropout           0.1            {args.dropout}            ← 3x stronger dropout!")
    print(f"  fc_dropout        (missing)      {args.fc_dropout}            ← Added regularization!")
    print("=" * 100)
    print()
    
    num_patches = int((args.seq_len - args.patch_len) / args.stride + 2)
    print(f"Sequence length: {args.seq_len} timesteps")
    print(f"Number of patches: {num_patches} (vs 90 patches at seq_len=720)")
    print(f"Patch ratio: {num_patches / 90:.1f}x more patches than 720")
    print(f"Expected runtime: ~8-12 hours on V100")
    print()
    
    setting = f'{args.model_id}_{args.data}_sl{args.seq_len}_pl{args.pred_len}'
    exp = Exp_Long_Term_Forecast(args)
    
    print("\n" + "=" * 100)
    print("TRAINING PHASE - This will take several hours...")
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
        
        # Load Phase 2A results for comparison
        phase2a_file = os.path.join(raft_root, 'experiments_patchtst/corrected/results/phase2a_patchtst_720_corrected.json')
        try:
            with open(phase2a_file, 'r') as f:
                phase2a_results = json.load(f)
            mse_720 = phase2a_results['test_mse']
        except:
            mse_720 = None
        
        print("\n" + "=" * 100)
        print("🚨 CRITICAL RESULTS - PHASE 2B")
        print("=" * 100)
        print(f"Test MSE:       {mse:.6f}")
        print(f"Test MAE:       {mae:.6f}")
        print(f"Training Time:  {training_time/60:.1f} minutes")
        print()
        print("=" * 100)
        print("FULL COMPARISON")
        print("=" * 100)
        print(f"  RAFT-720:              0.379 MSE")
        if mse_720:
            print(f"  PatchTST-720:          {mse_720:.3f} MSE (corrected)")
        print(f"  PatchTST-3000:         {mse:.3f} MSE (corrected)")
        print()
        
        # Determine scenario
        if mse < 0.45:
            print("=" * 100)
            print("📊 SCENARIO A: ORIGINAL PAPER IS CORRECT")
            print("=" * 100)
            print(f"  MSE {mse:.3f} < 0.45 → PatchTST handles long context well")
            print("  → Our original degradation was due to configuration errors")
            print("  → Paper needs MAJOR revision:")
            print("     • Remove claims about PatchTST degradation")
            print("     • Focus on 'RAFT outperforms PatchTST across all contexts'")
            print("     • Acknowledge configuration issues in methodology")
            scenario = "ORIGINAL_PAPER_CORRECT"
            recommendation = "MAJOR_REVISION_REQUIRED"
            
            if mse_720:
                if mse < mse_720:
                    print(f"  → PatchTST IMPROVES {((mse_720 - mse) / mse_720 * 100):.1f}% from 720→3000")
                    print("  → Matches original PatchTST paper findings")
                else:
                    print(f"  → Mild degradation {((mse - mse_720) / mse_720 * 100):.1f}% but still competitive")
        
        elif mse > 0.50:
            print("=" * 100)
            print("✅ SCENARIO B: OUR FINDING IS VALIDATED!")
            print("=" * 100)
            print(f"  MSE {mse:.3f} > 0.50 → PatchTST DEGRADES at long context")
            print("  → Even with CORRECT configuration, degradation persists!")
            print("  → NOVEL FINDING: First to show PatchTST fails at seq_len > 2000")
            print("  → Paper is VALID - proceed with current framing")
            scenario = "OUR_FINDING_VALIDATED"
            recommendation = "PROCEED_WITH_SUBMISSION"
            
            if mse_720:
                degradation = (mse - mse_720) / mse_720 * 100
                print(f"  → Degradation: {degradation:.1f}% from 720→3000")
                print(f"  → Even stronger than vanilla Transformer's degradation")
        
        else:
            print("=" * 100)
            print("⚠️  AMBIGUOUS RESULTS")
            print("=" * 100)
            print(f"  MSE {mse:.3f} is in gray zone (0.45-0.50)")
            print("  → Neither clearly validates nor refutes our claim")
            print("  → Need additional analysis:")
            print("     • Run Phase 3 (cross-domain) to check consistency")
            print("     • Examine training curves for instability")
            print("     • Consider larger ensemble (multiple runs)")
            scenario = "AMBIGUOUS"
            recommendation = "RUN_PHASE3_BEFORE_DECISION"
        
        print("=" * 100)
        
        # Save results
        results = {
            'experiment_name': 'Phase 2B - PatchTST-3000 (Corrected) - CRITICAL TEST',
            'phase': '2B',
            'purpose': 'Test if PatchTST degrades with longer context (corrected config)',
            'model': 'PatchTST',
            'seq_len': 3000,
            'test_mse': float(mse),
            'test_mae': float(mae),
            'training_time_minutes': round(training_time / 60, 1),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scenario': scenario,
            'recommendation': recommendation,
            'comparison': {
                'RAFT_720': 0.379,
                'PatchTST_720_corrected': float(mse_720) if mse_720 else None,
                'PatchTST_3000_corrected': float(mse)
            },
            'degradation_analysis': {
                'vs_raft': f"{((mse - 0.379) / 0.379 * 100):.1f}%",
                'vs_720': f"{((mse - mse_720) / mse_720 * 100):.1f}%" if mse_720 else None
            },
            'configuration': {
                'epochs': args.train_epochs,
                'batch_size': args.batch_size,
                'lradj': args.lradj,
                'd_model': args.d_model,
                'dropout': args.dropout
            },
            'next_steps': {
                'if_scenario_a': 'Major paper revision - remove degradation claims',
                'if_scenario_b': 'Proceed to Phase 3 for robustness, then submit',
                'if_ambiguous': 'Run Phase 3 cross-domain experiments'
            }
        }
        
        results_dir = os.path.join(raft_root, 'experiments_patchtst/corrected/results')
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = os.path.join(results_dir, 'phase2b_patchtst_3000_corrected_CRITICAL.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        print("\n" + "=" * 100)
        print("NEXT STEPS")
        print("=" * 100)
        if scenario == "OUR_FINDING_VALIDATED":
            print("✅ Run Phase 3 (cross-domain) to strengthen findings")
            print("✅ Prepare paper for submission with validated claims")
        elif scenario == "ORIGINAL_PAPER_CORRECT":
            print("⚠️  Begin major paper revision")
            print("⚠️  Reframe contribution as 'RAFT vs PatchTST comparison'")
        else:
            print("⚠️  Run Phase 3 experiments before making decision")
        print("=" * 100)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    run_experiment()
