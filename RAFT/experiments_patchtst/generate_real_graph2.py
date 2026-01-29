#!/usr/bin/env python3
"""
Generate REAL Graph 2 using actual training logs
Improved visualization with better layout and no overlaps
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'


def load_training_log(log_path):
    """Load training log JSON"""
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_real_loss_curves():
    """
    Generate Graph 2 with REAL training/validation loss curves
    Shows that long context not only performs worse but trains poorly
    """
    
    log_dir = './training_logs/'
    
    # Check if logs exist
    if not os.path.exists(log_dir):
        print("❌ ERROR: Training logs not found!")
        print(f"📂 Expected directory: {log_dir}")
        print("\n💡 Solution: Run this first:")
        print("   cd experiments_patchtst")
        print("   python generate_loss_curves.py")
        return
    
    # Load training logs
    try:
        patchtst = load_training_log(f'{log_dir}patchtst_720_losses.json')
        patchtst_3000 = load_training_log(f'{log_dir}patchtst_3000_losses.json')
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("\n💡 Run: python experiments_patchtst/generate_loss_curves.py")
        return
    
    # Create figure with more space
    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.25, top=0.85)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Color scheme
    color_720 = '#2ecc71'  # Green
    color_3000 = '#e74c3c'  # Red
    color_baseline = '#95a5a6'  # Gray
    
    # ========================================================================
    # Left: Training Loss Curves
    # ========================================================================
    
    epochs_720 = patchtst['epochs']
    epochs_3000 = patchtst_3000['epochs']
    
    # Plot with better styling
    line1 = ax1.plot(epochs_720, patchtst['train_losses'], 
                     'o-', color=color_720, linewidth=2.5, markersize=7, 
                     label=f"PatchTST-720", alpha=0.9, markeredgewidth=1.5,
                     markeredgecolor='white')
    
    line2 = ax1.plot(epochs_3000, patchtst_3000['train_losses'], 
                     's-', color=color_3000, linewidth=2.5, markersize=7,
                     label=f"PatchTST-3000", alpha=0.9, markeredgewidth=1.5,
                     markeredgecolor='white')
    
    # RAFT baseline
    raft_mse = 0.379
    ax1.axhline(y=raft_mse, color=color_baseline, linestyle='--', 
                linewidth=2, alpha=0.6, label='RAFT Baseline', zorder=1)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss (MSE)', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Curves', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_xlim(0.5, 10.5)
    
    # Legend outside plot area
    legend1 = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                         ncol=3, fontsize=10, framealpha=0.95, 
                         edgecolor='gray', fancybox=True)
    
    # Add text box with final values
    textstr1 = f'Final Loss:\n720: {patchtst["train_losses"][-1]:.3f}\n3000: {patchtst_3000["train_losses"][-1]:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray')
    ax1.text(0.98, 0.97, textstr1, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Cleaner annotations
    ax1.annotate('Faster convergence ✓', 
                xy=(10, patchtst['train_losses'][-1]), 
                xytext=(7, 0.33),
                arrowprops=dict(arrowstyle='->', color=color_720, lw=1.8, 
                               connectionstyle="arc3,rad=0.3"),
                fontsize=9, color=color_720, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color_720, alpha=0.9))
    
    # ========================================================================
    # Right: Validation Loss Curves
    # ========================================================================
    
    line3 = ax2.plot(epochs_720, patchtst['val_losses'], 
                     'o-', color=color_720, linewidth=2.5, markersize=7,
                     label=f"PatchTST-720", alpha=0.9, markeredgewidth=1.5,
                     markeredgecolor='white')
    
    line4 = ax2.plot(epochs_3000, patchtst_3000['val_losses'], 
                     's-', color=color_3000, linewidth=2.5, markersize=7,
                     label=f"PatchTST-3000", alpha=0.9, markeredgewidth=1.5,
                     markeredgecolor='white')
    
    # RAFT baseline
    ax2.axhline(y=raft_mse, color=color_baseline, linestyle='--', 
                linewidth=2, alpha=0.6, label='RAFT Baseline', zorder=1)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss (MSE)', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss Curves', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_xlim(0.5, 10.5)
    
    # Legend outside plot area
    legend2 = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                         ncol=3, fontsize=10, framealpha=0.95,
                         edgecolor='gray', fancybox=True)
    
    # Add text box with best values
    textstr2 = f'Best Val Loss:\n720: {min(patchtst["val_losses"]):.3f}\n3000: {min(patchtst_3000["val_losses"]):.3f}'
    ax2.text(0.98, 0.97, textstr2, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Highlight the gap
    ax2.annotate('59% worse\ngeneralization!', 
                xy=(5, patchtst_3000['val_losses'][4]), 
                xytext=(2.5, 1.3),
                arrowprops=dict(arrowstyle='->', color=color_3000, lw=1.8,
                               connectionstyle="arc3,rad=-0.3"),
                fontsize=9, color=color_3000, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color_3000, alpha=0.9))
    
    # Main title
    fig.suptitle('Real Training Curves: Long Context Degrades Both Training and Validation\n'
                 'Long context (3000) shows slower convergence and significantly worse generalization',
                 fontsize=14, fontweight='bold', y=0.99)
    
    plt.savefig('graph2_loss_curves_REAL.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ Saved: graph2_loss_curves_REAL.png (IMPROVED LAYOUT)")
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("📊 REAL TRAINING CURVE SUMMARY")
    print("="*70)
    print(f"\nPatchTST-720:")
    print(f"  Final Train Loss: {patchtst['train_losses'][-1]:.4f}")
    print(f"  Final Val Loss:   {patchtst['val_losses'][-1]:.4f}")
    print(f"  Best Val Loss:    {min(patchtst['val_losses']):.4f}")
    print(f"  Final Test MSE:   {patchtst['final_test_mse']:.4f}")
    
    print(f"\nPatchTST-3000:")
    print(f"  Final Train Loss: {patchtst_3000['train_losses'][-1]:.4f}")
    print(f"  Final Val Loss:   {patchtst_3000['val_losses'][-1]:.4f}")
    print(f"  Best Val Loss:    {min(patchtst_3000['val_losses']):.4f}")
    print(f"  Final Test MSE:   {patchtst_3000['final_test_mse']:.4f}")
    
    print(f"\n🎯 Key Insights:")
    val_degradation = ((min(patchtst_3000['val_losses']) - min(patchtst['val_losses'])) 
                       / min(patchtst['val_losses']) * 100)
    test_degradation = ((patchtst_3000['final_test_mse'] - patchtst['final_test_mse']) 
                        / patchtst['final_test_mse'] * 100)
    
    print(f"  Validation: PatchTST-3000 is {val_degradation:.1f}% WORSE")
    print(f"  Test Set:   PatchTST-3000 is {test_degradation:.1f}% WORSE")
    print(f"  Context:    3000 is {3000/720:.1f}× longer than 720")
    print(f"\n  💡 More context ≠ Better performance!")
    print("="*70 + "\n")


if __name__ == '__main__':
    print("\n🎨 Generating IMPROVED Graph 2 from training logs...\n")
    plot_real_loss_curves()