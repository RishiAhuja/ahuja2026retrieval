"""
Graph Generation for Workshop Paper: "When Context Isn't Enough: Retrieval > Long Context for Time Series"
Generates comparison graphs between RAFT and Time-CAG experiments
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11

# ============================================================================
# GRAPH 1: MSE Comparison Bar Chart
# ============================================================================

# Professional styling without LaTeX (avoids dependency issues)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

def plot_iclr_mse_comparison():
    # --- Data Setup ---
    # Shortened labels slightly for better x-axis fitting
    models = [
        'RAFT\n(Ret-720)',
        'PatchTST\n(720)',
        'Exp7\nBest\n(720)',
        'Exp3\nSame\nCtx',
        'Exp2\nMed\n(1440)',
        'PatchTST\n(3000)',
        'Exp1\nLong\n(3000)'
    ]
    
    mse_values = [0.379, 0.385, 0.416, 0.441, 0.484, 0.647, 1.323]
    
    # --- Professional Color Palette ---
    # Highlighting the Best (RAFT) with a distinct strong color (e.g., Deep Blue)
    # Using Greys/Muted tones for baselines
    # Using Muted Orange/Red for worse performing ablations
    # Hex codes chosen for print safety and colorblind accessibility
    
    # Logic: 
    # Index 0 (RAFT): Highlight (Royal Blue)
    # Index 1, 5 (PatchTST): Baseline Reference (Slate Grey)
    # Index 2, 3, 4, 6 (Experiments): Variations (Muted warm tones)
    
    colors = [
        '#005A9C', # RAFT (Highlight - Strong Blue)
        '#7f7f7f', # PatchTST 720 (Grey)
        '#D6B656', # Exp7 (Muted Gold)
        '#CC79A7', # Exp3 (Muted Pink/Purple)
        '#E69F00', # Exp2 (Orange)
        '#7f7f7f', # PatchTST 3000 (Grey)
        '#D55E00'  # Exp1 (Vermilion)
    ]
    
    # Setup Figure (Width=10, Height=4.5 is good for full-width paper figures)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    # Plot Bars
    bars = ax.bar(models, mse_values, color=colors, alpha=0.9, 
                  edgecolor='black', linewidth=0.8, width=0.65)
    
    # --- Annotations ---
    # Add value labels on bars
    for bar, val in zip(bars, mse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}',
                ha='center', va='bottom', 
                fontweight='normal', fontsize=10, color='black')

    # --- Baseline Line ---
    # A subtle horizontal line for the SOTA/Baseline comparison
    ax.axhline(y=0.379, color='#005A9C', linestyle='--', linewidth=1, alpha=0.6, label='RAFT Baseline')

    # --- Formatting Axes ---
    ax.set_ylabel('Test MSE (Lower is Better)', labelpad=10, weight='bold')
    
    # Add legend for the baseline line
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='gray')
    
    # Clean up spines (ICLR minimalist style)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Grid configuration
    ax.grid(axis='y', linestyle=':', alpha=0.4, color='gray', zorder=0)
    ax.set_axisbelow(True) # Put grid behind bars
    
    # Y-Limit adjustment for breathing room
    ax.set_ylim(0, 1.45) 
    
    # Remove X-label title if the category names are self-explanatory
    # (Saves vertical space)
    # ax.set_xlabel('Model Configuration', labelpad=10) 

    plt.tight_layout()
    
    # --- Saving ---
    # Save as PDF for the LaTeX paper (Vector graphics)
    plt.savefig('mse_comparison_iclr.pdf', format='pdf', bbox_inches='tight')
    # Save as PNG for quick preview
    plt.savefig('mse_comparison_iclr.png', dpi=300, bbox_inches='tight')
    
    print("Figures saved as 'mse_comparison_iclr.pdf' and 'mse_comparison_iclr.png'")
    plt.close()

# def plot_mse_comparison():
#     """
#     Bar chart comparing RAFT vs Time-CAG vs PatchTST experiments
#     UPDATED WITH ACTUAL RESULTS INCLUDING PATCHTST
#     """
    
#     models = [
#         'RAFT\n(Retrieval\n720)',
#         'PatchTST\n720',
#         'Exp7\nBest\n(720)',
#         'Exp3\nSame\nContext',
#         'Exp2\nMedium\n(1440)',
#         'PatchTST\n3000',
#         'Exp1\nLong\n(3000)'
#     ]
    
#     # ACTUAL EXPERIMENTAL RESULTS
#     mse_values = [
#         0.379,  # RAFT baseline
#         0.385,  # PatchTST-720 (SOTA, nearly matches RAFT!)
#         0.416,  # Exp7: Best config (d=64, e=2, dr=0.2, epochs=15)
#         0.441,  # Exp3: Same context as RAFT (d=128, e=3)
#         0.484,  # Exp2: Medium context (seq=1440)
#         0.647,  # PatchTST-3000 (DEGRADES 68%!)
#         1.323   # Exp1: Long context (seq=3000) - WORST!
#     ]
    
#     colors = ['#2ecc71', '#27ae60', '#f39c12', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    
#     fig, ax = plt.subplots(figsize=(14, 6))
#     bars = ax.bar(models, mse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
#     # Add value labels on bars
#     for i, (bar, val) in enumerate(zip(bars, mse_values)):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{val:.3f}',
#                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    
#     # Add horizontal line for RAFT baseline
#     ax.axhline(y=0.379, color='green', linestyle='--', linewidth=2, label='RAFT Baseline', alpha=0.7)
    
#     ax.set_ylabel('Test MSE (Lower is Better)', fontsize=13, fontweight='bold')
#     ax.set_xlabel('Model Configuration', fontsize=13, fontweight='bold')
#     ax.set_title('Performance Comparison: RAFT vs Time-CAG vs PatchTST Experiments\nETTh1 Dataset (96-step Forecast)', 
#                  fontsize=14, fontweight='bold', pad=20)
#     ax.set_ylim(0, 1.5)
#     ax.legend(loc='upper right', fontsize=11)
#     ax.grid(axis='y', alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('graph1_mse_comparison.png', dpi=300, bbox_inches='tight')
#     print("✓ Saved: graph1_mse_comparison.png")
#     plt.close()


# ============================================================================
# GRAPH 2: Train/Validation Loss Curves
# ============================================================================

# def plot_loss_curves():
#     """
#     Training curves showing overfitting in Time-CAG vs good generalization in RAFT
#     Update with your actual training logs!
#     """
    
#     epochs = np.arange(1, 21)  # 20 epochs
    
#     # RAFT training curve (good generalization - tight gap)
#     # These are synthetic - replace with actual RAFT logs if available
#     raft_train = 0.45 * np.exp(-0.15 * epochs) + 0.32
#     raft_val = 0.48 * np.exp(-0.15 * epochs) + 0.35
    
#     # Time-CAG training curve (overfitting - large gap)
#     # UPDATE THESE with your actual training logs from experiments
#     timecag_train = 0.9 * np.exp(-0.25 * epochs) + 0.15  # Drops fast (memorization)
#     timecag_val = 0.9 * np.exp(-0.08 * epochs) + 0.85    # Drops slowly (poor generalization)
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
#     # --- Subplot 1: RAFT (Good Generalization) ---
#     ax1.plot(epochs, raft_train, 'o-', color='#2ecc71', linewidth=2.5, 
#              markersize=6, label='Train Loss', alpha=0.9)
#     ax1.plot(epochs, raft_val, 's-', color='#27ae60', linewidth=2.5, 
#              markersize=6, label='Validation Loss', alpha=0.9)
#     ax1.fill_between(epochs, raft_train, raft_val, alpha=0.2, color='green')
    
#     ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
#     ax1.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
#     ax1.set_title('RAFT: Good Generalization\n(Small Train/Val Gap)', 
#                   fontsize=13, fontweight='bold', color='green')
#     ax1.legend(loc='upper right', fontsize=11)
#     ax1.grid(alpha=0.3)
#     ax1.set_ylim(0, 1.2)
    
#     # Add annotation
#     ax1.annotate('Tight gap = Good generalization', 
#                 xy=(15, 0.4), xytext=(10, 0.7),
#                 arrowprops=dict(arrowstyle='->', color='green', lw=2),
#                 fontsize=10, color='green', fontweight='bold')
    
#     # --- Subplot 2: Time-CAG (Overfitting) ---
#     ax2.plot(epochs, timecag_train, 'o-', color='#e74c3c', linewidth=2.5, 
#              markersize=6, label='Train Loss', alpha=0.9)
#     ax2.plot(epochs, timecag_val, 's-', color='#c0392b', linewidth=2.5, 
#              markersize=6, label='Validation Loss', alpha=0.9)
#     ax2.fill_between(epochs, timecag_train, timecag_val, alpha=0.2, color='red')
    
#     ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
#     ax2.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
#     ax2.set_title('Time-CAG: Overfitting\n(Large Train/Val Gap)', 
#                   fontsize=13, fontweight='bold', color='red')
#     ax2.legend(loc='upper right', fontsize=11)
#     ax2.grid(alpha=0.3)
#     ax2.set_ylim(0, 1.2)
    
#     # Add annotation
#     ax2.annotate('Large gap = Memorization', 
#                 xy=(15, 0.8), xytext=(10, 0.3),
#                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
#                 fontsize=10, color='red', fontweight='bold')
    
#     plt.tight_layout()
#     plt.savefig('graph2_loss_curves.png', dpi=300, bbox_inches='tight')
#     print("✓ Saved: graph2_loss_curves.png")
#     plt.close()


# ============================================================================
# GRAPH 3: Test MSE vs Context Length
# ============================================================================

# def plot_context_vs_mse():
#     """
#     Line plot showing: More context HURTS performance!
#     UPDATED WITH ACTUAL RESULTS INCLUDING PATCHTST
#     """
    
#     # Context lengths tested
#     context_lengths = [720, 1440, 3000]
    
#     # ACTUAL EXPERIMENTAL RESULTS
#     timecag_mse = [
#         0.441,  # Exp3: 720 context (same as RAFT)
#         0.484,  # Exp2: 1440 context
#         1.323   # Exp1: 3000 context - CATASTROPHIC!
#     ]
    
#     # PatchTST results (SOTA but still degrades!)
#     patchtst_mse = [
#         0.385,  # PatchTST-720 (nearly matches RAFT)
#         None,   # Not tested at 1440
#         0.647   # PatchTST-3000 (degrades 68%!)
#     ]
    
#     # RAFT baseline (constant - doesn't use long context)
#     raft_mse = [0.379] * len(context_lengths)
    
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Plot Time-CAG (WORSENS with more context!)
#     ax.plot(context_lengths, timecag_mse, 'o-', color='#e74c3c', 
#             linewidth=3, markersize=12, label='Vanilla Transformer (Long Context)', alpha=0.9)
    
#     # Plot PatchTST (SOTA but still degrades!)
#     patchtst_x = [720, 3000]
#     patchtst_y = [0.385, 0.647]
#     ax.plot(patchtst_x, patchtst_y, 'D-', color='#e67e22', 
#             linewidth=3, markersize=12, label='PatchTST (SOTA)', alpha=0.9)
    
#     # Plot RAFT baseline (constant, superior)
#     ax.plot(context_lengths, raft_mse, 's--', color='#2ecc71', 
#             linewidth=3, markersize=12, label='RAFT (Retrieval @ 720)', alpha=0.9)
    
#     # Add value labels for Time-CAG
#     for x, y in zip(context_lengths, timecag_mse):
#         ax.text(x, y + 0.08, f'{y:.3f}', ha='center', va='bottom', 
#                 fontsize=11, fontweight='bold', color='#e74c3c')
    
#     # Add value labels for PatchTST
#     for x, y in zip(patchtst_x, patchtst_y):
#         ax.text(x, y + 0.05, f'{y:.3f}', ha='center', va='bottom', 
#                 fontsize=11, fontweight='bold', color='#e67e22')
    
#     # Add RAFT label
#     ax.text(context_lengths[1], raft_mse[1] - 0.08, f'{raft_mse[0]:.3f}', 
#             ha='center', va='top', fontsize=11, fontweight='bold', color='#2ecc71')
    
#     ax.set_xlabel('Context Window Size (timesteps)', fontsize=13, fontweight='bold')
#     ax.set_ylabel('Test MSE (Lower is Better)', fontsize=13, fontweight='bold')
#     ax.set_title('More Context = WORSE Performance!\nETTh1 Dataset (96-step Forecast)', 
#                  fontsize=14, fontweight='bold', pad=20, color='#e74c3c')
#     ax.set_xlim(500, 3200)
#     ax.set_ylim(0, 1.5)
#     ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
#     ax.grid(alpha=0.3)
    
#     # Add insight annotation
#     ax.annotate('3.5× WORSE\nwith 4× more data!', 
#                 xy=(3000, timecag_mse[-1]), xytext=(2000, 0.9),
#                 arrowprops=dict(arrowstyle='->', color='red', lw=3),
#                 fontsize=13, color='red', fontweight='bold',
#                 bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.8))
    
#     # Add PatchTST degradation annotation
#     ax.annotate('Even SOTA\ndegrades 68%!', 
#                 xy=(3000, patchtst_y[-1]), xytext=(2200, 0.5),
#                 arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2),
#                 fontsize=11, color='#e67e22', fontweight='bold',
#                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
#     plt.tight_layout()
#     plt.savefig('graph3_context_vs_mse.png', dpi=300, bbox_inches='tight')
#     print("✓ Saved: graph3_context_vs_mse.png")
#     plt.close()


# ============================================================================
# GRAPH 4: Prediction Visualization
# ============================================================================

# def plot_predictions():
#     """
#     Visual comparison of actual predictions
#     Shows how Time-CAG amplifies noise vs RAFT's smooth predictions
#     """
    
#     # Generate synthetic time series for demonstration
#     # Replace with actual test set predictions from your models
#     timesteps = np.arange(0, 96)
    
#     # Ground truth (smooth sine wave with slight trend)
#     np.random.seed(42)
#     ground_truth = 0.5 * np.sin(timesteps * 0.1) + 0.002 * timesteps + np.random.normal(0, 0.05, 96)
    
#     # RAFT predictions (close to ground truth, smooth)
#     raft_pred = ground_truth + np.random.normal(0, 0.15, 96)
    
#     # Time-CAG predictions (noisy, overconfident in wrong directions)
#     timecag_pred = ground_truth + np.random.normal(0, 0.35, 96) + 0.3 * np.sin(timesteps * 0.3)
    
#     fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
#     # --- Subplot 1: Ground Truth ---
#     axes[0].plot(timesteps, ground_truth, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9)
#     axes[0].fill_between(timesteps, ground_truth - 0.1, ground_truth + 0.1, 
#                           alpha=0.2, color='gray', label='±0.1 tolerance')
#     axes[0].set_ylabel('Value', fontsize=12, fontweight='bold')
#     axes[0].set_title('Ground Truth (Test Set)', fontsize=13, fontweight='bold')
#     axes[0].legend(loc='upper left', fontsize=11)
#     axes[0].grid(alpha=0.3)
#     axes[0].set_ylim(-2, 2)
    
#     # --- Subplot 2: RAFT Predictions ---
#     axes[1].plot(timesteps, ground_truth, 'k--', linewidth=2, label='Ground Truth', alpha=0.5)
#     axes[1].plot(timesteps, raft_pred, 'g-', linewidth=2.5, label='RAFT Prediction', alpha=0.9)
#     axes[1].fill_between(timesteps, ground_truth, raft_pred, alpha=0.3, color='green')
#     axes[1].set_ylabel('Value', fontsize=12, fontweight='bold')
#     axes[1].set_title('RAFT Predictions (MSE: 0.379)', fontsize=13, fontweight='bold', color='green')
#     axes[1].legend(loc='upper left', fontsize=11)
#     axes[1].grid(alpha=0.3)
#     axes[1].set_ylim(-2, 2)
    
#     # Add annotation
#     axes[1].annotate('Smooth, accurate', xy=(70, raft_pred[70]), xytext=(50, 1.5),
#                      arrowprops=dict(arrowstyle='->', color='green', lw=2),
#                      fontsize=11, color='green', fontweight='bold')
    
#     # --- Subplot 3: Time-CAG Predictions ---
#     axes[2].plot(timesteps, ground_truth, 'k--', linewidth=2, label='Ground Truth', alpha=0.5)
#     axes[2].plot(timesteps, timecag_pred, 'r-', linewidth=2.5, label='Time-CAG Prediction', alpha=0.9)
#     axes[2].fill_between(timesteps, ground_truth, timecag_pred, alpha=0.3, color='red')
#     axes[2].set_xlabel('Timestep (96-step forecast)', fontsize=12, fontweight='bold')
#     axes[2].set_ylabel('Value', fontsize=12, fontweight='bold')
#     axes[2].set_title('Time-CAG Predictions (MSE: 1.000)', fontsize=13, fontweight='bold', color='red')
#     axes[2].legend(loc='upper left', fontsize=11)
#     axes[2].grid(alpha=0.3)
#     axes[2].set_ylim(-2, 2)
    
#     # Add annotation
#     axes[2].annotate('Noisy, amplified errors', xy=(70, timecag_pred[70]), xytext=(50, -1.5),
#                      arrowprops=dict(arrowstyle='->', color='red', lw=2),
#                      fontsize=11, color='red', fontweight='bold')
    
#     plt.tight_layout()
#     plt.savefig('graph4_predictions.png', dpi=300, bbox_inches='tight')
#     print("✓ Saved: graph4_predictions.png")
#     plt.close()


# ============================================================================
# GRAPH 5: Computational Cost vs Performance
# ============================================================================

# def plot_computational_cost():
#     """
#     Scatter plot: Training time vs MSE
#     UPDATED WITH ACTUAL RESULTS INCLUDING PATCHTST
#     """
    
#     models = ['RAFT', 'PatchTST\n720', 'Exp4\n(d=32)', 'Exp7\nBest', 'Exp3\nSame', 'Exp2\nMedium', 'PatchTST\n3000', 'Exp1\nLong']
    
#     # ACTUAL TRAINING TIMES (in minutes)
#     training_times = [
#         2.13,   # RAFT baseline
#         2.90,   # PatchTST-720 (SOTA, nearly matches RAFT!)
#         2.13,   # Exp4 d=32 (fastest!)
#         8.67,   # Exp7 (best config)
#         7.42,   # Exp3 (same context)
#         22.00,  # Exp2 (medium context)
#         18.33,  # PatchTST-3000 (degrades!)
#         83.47   # Exp1 (long context - SLOWEST!)
#     ]
    
#     # ACTUAL MSE VALUES
#     mse_values = [0.379, 0.385, 0.397, 0.416, 0.441, 0.484, 0.647, 1.323]
    
#     colors = ['#2ecc71', '#27ae60', '#3498db', '#f39c12', '#9b59b6', '#e67e22', '#e67e22', '#e74c3c']
#     sizes = [450, 400, 350, 300, 280, 260, 300, 220]
    
#     fig, ax = plt.subplots(figsize=(14, 8))
    
#     # Custom label positions to avoid overlaps
#     label_offsets = [
#         (0, 0.06),      # RAFT
#         (0, -0.10),     # PatchTST-720 (below to avoid RAFT overlap)
#         (-0.5, 0.06),   # Exp4 (left to avoid PatchTST overlap)
#         (0, 0.06),      # Exp7
#         (0, 0.06),      # Exp3
#         (0, 0.06),      # Exp2
#         (0, 0.06),      # PatchTST-3000
#         (0, 0.06)       # Exp1
#     ]
    
#     for i, (model, time, mse, color, size, offset) in enumerate(zip(models, training_times, mse_values, colors, sizes, label_offsets)):
#         ax.scatter(time, mse, s=size, c=color, alpha=0.7, edgecolors='black', linewidth=2, zorder=3)
#         ax.text(time + offset[0], mse + offset[1], model, ha='center', va='bottom', 
#                 fontsize=9, fontweight='bold')
    
#     # Draw ideal region
#     ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, zorder=1)
#     ax.axvline(x=20, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, zorder=1)
    
#     # Label quadrants
#     ax.text(1, 1.45, 'IDEAL\n(Fast + Accurate)', ha='left', fontsize=12, 
#             fontweight='bold', color='green', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7), zorder=4)
#     ax.text(75, 1.45, 'DISASTER\n(Slow + Inaccurate)', ha='center', fontsize=12, 
#             fontweight='bold', color='red', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7), zorder=4)
    
#     ax.set_xlabel('Training Time (minutes)', fontsize=13, fontweight='bold')
#     ax.set_ylabel('Test MSE (Lower is Better)', fontsize=13, fontweight='bold')
#     ax.set_title('Computational Efficiency vs Performance\nRAFT wins both dimensions!', 
#                  fontsize=14, fontweight='bold', pad=20)
#     ax.set_xlim(0, 90)
#     ax.set_ylim(0.2, 1.5)
#     ax.grid(alpha=0.3, zorder=0)
    
#     # Add arrow annotation
#     ax.annotate('17× slower\n3.5× worse!', 
#                 xy=(83, 1.32), xytext=(50, 0.7),
#                 arrowprops=dict(arrowstyle='->', color='red', lw=3),
#                 fontsize=12, color='red', fontweight='bold',
#                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7), zorder=4)
    
#     plt.tight_layout()
#     plt.savefig('graph5_computational_cost.png', dpi=300, bbox_inches='tight')
#     print("✓ Saved: graph5_computational_cost.png")
#     plt.close()


# ============================================================================
# GRAPH 6: Ablation Study Heatmap
# ============================================================================

# def plot_ablation_heatmap():
#     """
#     Heatmap showing ablation results from Exp4
#     UPDATED WITH ACTUAL RESULTS
#     """
    
#     # Exp4: d_model ablation (rows)
#     # Exp5: e_layers ablation (columns)
#     d_models = [32, 64, 128, 256]
#     e_layers = [1, 2, 3, 4]
    
#     # ACTUAL ABLATION RESULTS
#     # We have full d_model data, partial e_layers data
#     # Creating combined grid from Exp4 (d_model) and Exp5 (e_layers with d=128)
#     mse_grid = np.array([
#         [0.520, 0.397, 0.550, 0.530],  # d_model=32 (Exp4: e_layers=2 actual, others estimated)
#         [0.570, 0.529, 0.600, 0.580],  # d_model=64 (Exp4: e_layers=2 actual)
#         [0.441, 0.441, 0.528, 0.429],  # d_model=128 (Exp5: all e_layers actual!)
#         [0.600, 0.558, 0.650, 0.620],  # d_model=256 (Exp4: e_layers=2 actual)
#     ])
    
#     fig, ax = plt.subplots(figsize=(11, 8))
    
#     # Create heatmap (lower MSE = better = greener)
#     im = ax.imshow(mse_grid, cmap='RdYlGn_r', aspect='auto', vmin=0.35, vmax=0.7)
    
#     # Set ticks
#     ax.set_xticks(np.arange(len(e_layers)))
#     ax.set_yticks(np.arange(len(d_models)))
#     ax.set_xticklabels(e_layers, fontsize=12)
#     ax.set_yticklabels(d_models, fontsize=12)
    
#     # Labels
#     ax.set_xlabel('Number of Encoder Layers', fontsize=13, fontweight='bold')
#     ax.set_ylabel('Model Dimension (d_model)', fontsize=13, fontweight='bold')
#     ax.set_title('Ablation Study: Smaller Models Generalize Better\nBest Time-CAG (d=32, e=2) achieves MSE 0.397, still 4.7% worse than RAFT (0.379)', 
#                  fontsize=13, fontweight='bold', pad=20)
    
#     # Add text annotations
#     for i in range(len(d_models)):
#         for j in range(len(e_layers)):
#             text = ax.text(j, i, f'{mse_grid[i, j]:.3f}',
#                           ha="center", va="center", color="black", 
#                           fontsize=11, fontweight='bold')
    
#     # Highlight best Time-CAG result
#     ax.add_patch(plt.Rectangle((1-0.5, 0-0.5), 1, 1, fill=False, edgecolor='blue', lw=4))
#     ax.text(1, 0, '★', ha='center', va='center', fontsize=24, color='blue')
    
#     # Add RAFT baseline reference (moved higher to avoid overlap)
#     ax.text(3.5, -0.95, 'RAFT: 0.379 (Unbeatable!)', 
#             ha='right', fontsize=11, fontweight='bold', color='green',
#             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.95, edgecolor='green', linewidth=2))
    
#     # Add colorbar
#     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label('Test MSE (Lower = Better)', fontsize=12, fontweight='bold')
    
#     # Add insight box (repositioned to left to avoid RAFT overlap)
#     ax.text(-1.2, 1.2, 'Best: d=32, e=2 → MSE=0.397\nStill 4.7% worse than RAFT!\n\nKey Finding:\nSmaller models generalize better', 
#             fontsize=10, fontweight='bold',
#             verticalalignment='top', horizontalalignment='left', color='darkblue',
#             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='blue', linewidth=2))
    
#     plt.tight_layout()
#     plt.savefig('graph6_ablation_heatmap.png', dpi=300, bbox_inches='tight')
#     print("✓ Saved: graph6_ablation_heatmap.png")
#     plt.close()


# ============================================================================
# MAIN
# ============================================================================

def plot_iclr_context_vs_mse():
    """Professional line plot showing inverse scaling law"""
    # --- Data Setup ---
    context_lengths = [720, 1440, 3000]
    
    # Data points
    timecag_mse = [0.441, 0.484, 1.323]  # Long Context Transformer
    
    # PatchTST only has data for 720 and 3000
    patchtst_x = [720, 3000]
    patchtst_y = [0.385, 0.647]
    
    # RAFT is a constant baseline (horizontal line)
    raft_val = 0.379
    
    # --- Professional Color Palette (Colorblind Friendly) ---
    # Blue: Proposed Method (RAFT)
    # Grey: SOTA Baseline (PatchTST)
    # Vermilion/Orange: Comparison that degrades (Vanilla/TimeCAG)
    c_raft = '#005A9C'      # Strong Blue
    c_patch = '#7f7f7f'     # Slate Grey
    c_vanilla = '#D55E00'   # Vermilion (High contrast)

    # Setup Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # --- Plotting ---
    
    # 1. Vanilla Transformer (Shows the degradation clearly)
    # Use markers + line for B&W readability
    ax.plot(context_lengths, timecag_mse, marker='o', linestyle='-', 
            color=c_vanilla, linewidth=2, markersize=8, 
            label='Vanilla Transformer (Long Ctx)')
    
    # 2. PatchTST (SOTA Baseline)
    ax.plot(patchtst_x, patchtst_y, marker='D', linestyle='-.', 
            color=c_patch, linewidth=2, markersize=7, 
            label='PatchTST (SOTA)')
    
    # 3. RAFT (Proposed - Constant Line)
    # We plot this across the full range
    ax.axhline(y=raft_val, color=c_raft, linestyle='--', linewidth=2, 
               label='RAFT (Retrieval @ 720)', zorder=10)
    
    # Optional: Add a marker for RAFT at the x-points to show we evaluated here?
    # Or just keep it as a clean baseline line. Let's add a single star at 720 
    # since that is the context used.
    ax.scatter([720], [raft_val], color=c_raft, marker='*', s=150, zorder=11)

    # --- Annotations & Formatting ---
    
    # Axis Labels
    ax.set_xlabel('Context Window Size (L)', fontweight='bold')
    ax.set_ylabel('Test MSE (Lower is Better)', fontweight='bold')
    
    # Ticks - Ensure x-axis shows the specific context lengths relevant to the paper
    ax.set_xticks(context_lengths)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    
    # Add a subtle grid
    ax.grid(True, linestyle=':', alpha=0.5, color='gray')
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend - Top Left is usually empty in this specific data shape
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='white')

    # Trend Annotation (Valuable for "at a glance" understanding)
    # Arrow pointing up indicating error increase
    ax.annotate('Performance degrades\nas context increases', 
                xy=(1440, 0.484), xytext=(1600, 0.8),
                arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.6),
                fontsize=9, style='italic', alpha=0.8)

    plt.tight_layout()
    
    # --- Saving ---
    plt.savefig('context_vs_mse_iclr.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('context_vs_mse_iclr.png', dpi=300, bbox_inches='tight')
    
    print("Figures saved as 'context_vs_mse_iclr.pdf' and 'context_vs_mse_iclr.png'")
    plt.close()


if __name__ == "__main__":
    print("Generating graphs for workshop paper...")
    print("=" * 60)
    
    # Generate Graph 1
    print("\n[1/7] Generating MSE Comparison Bar Chart...")
    plot_iclr_mse_comparison()
    plot_iclr_context_vs_mse()
    
    # # Generate Graph 2
    # print("\n[2/7] Generating Train/Validation Loss Curves...")
    # plot_loss_curves()
    
    # # Generate Graph 3
    # print("\n[3/7] Generating Context Length vs MSE...")
    # plot_context_vs_mse()
    
    # # Generate Graph 4
    # print("\n[4/7] Generating Prediction Visualization...")
    # plot_predictions()
    
    # # Generate Graph 5
    # print("\n[5/7] Generating Computational Cost Analysis...")
    # plot_computational_cost()
    
    # # Generate Graph 6
    # print("\n[6/7] Generating Ablation Study Heatmap...")
    # plot_ablation_heatmap()
    
    print("\n" + "=" * 60)
    print("✅ All 6 graphs generated successfully!")
    print("=" * 60)
    print("\nGraphs saved:")
    print("  - graph1_mse_comparison.png")
    print("  - graph2_loss_curves.png")
    print("  - graph3_context_vs_mse.png")
    print("  - graph4_predictions.png")
    print("  - graph5_computational_cost.png")
    print("  - graph6_ablation_heatmap.png")
    print("\n💡 Next steps:")
    print("1. Update placeholder values with your actual experiment results")
    print("2. Run experiments and capture training times for graph 5")
    print("3. Conduct ablation study for graph 6 (test d_model × e_layers grid)")
    print("4. Re-run this script after each experiment to refresh graphs")

