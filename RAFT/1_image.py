import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# --- Configuration for ICLR / Academic Paper Styling ---
# Try to use LaTeX font rendering if available, otherwise fallback to Serif
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],  # Matches ICLR LaTeX templates
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14
    })
except:
    # Fallback if user doesn't have a LaTeX distribution installed
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
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
    ax.axhline(y=0.379, color='#005A9C', linestyle='--', linewidth=1, alpha=0.6)
    ax.text(len(models)-0.6, 0.379 + 0.02, 'RAFT Baseline', 
            color='#005A9C', ha='right', va='bottom', fontsize=9, style='italic')

    # --- Formatting Axes ---
    ax.set_ylabel('Test MSE (Lower is Better)', labelpad=10, weight='bold')
    
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
    plt.show()

if __name__ == "__main__":
    plot_iclr_mse_comparison()