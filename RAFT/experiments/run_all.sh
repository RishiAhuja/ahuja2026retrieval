#!/bin/bash
# Master script to run all 7 experiments sequentially
# Total estimated time: 8-12 hours

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "RUNNING ALL 7 EXPERIMENTS FOR WORKSHOP PAPER"
echo "════════════════════════════════════════════════════════════════"
echo "Start time: $(date)"
echo ""
echo "Estimated total runtime: 8-12 hours"
echo ""

# Detect environment and navigate to RAFT root
if [ -d "/content/RAFT" ]; then
    echo "🌐 Running in Google Colab"
    cd /content/RAFT
    EXPERIMENTS_DIR="/content/RAFT/experiments"
elif [ -d "../venv" ]; then
    echo "💻 Running locally (Mac)"
    cd "$(dirname "$0")/.."
    source venv/bin/activate
    EXPERIMENTS_DIR="./experiments"
else
    echo "⚠️  Unknown environment - assuming current directory"
    cd "$(dirname "$0")/.."
    EXPERIMENTS_DIR="./experiments"
fi

echo "Working directory: $(pwd)"
echo "Experiments directory: $EXPERIMENTS_DIR"
echo ""

# Verify all experiment files exist
echo "🔍 Verifying experiment files..."
missing_files=0
for exp in exp1_timecag_v1.py exp2_timecag_v2.py exp3_timecag_v3.py exp4_ablation_dmodel.py exp5_ablation_elayers.py exp6_ablation_dropout.py exp7_best_config.py; do
    if [ -f "$EXPERIMENTS_DIR/$exp" ]; then
        echo "  ✓ $exp"
    else
        echo "  ✗ MISSING: $exp"
        missing_files=$((missing_files + 1))
    fi
done

if [ $missing_files -gt 0 ]; then
    echo ""
    echo "❌ ERROR: $missing_files experiment file(s) missing!"
    echo "Please ensure all files are in the experiments/ directory."
    echo ""
    echo "Files should be:"
    echo "  - exp1_timecag_v1.py"
    echo "  - exp2_timecag_v2.py"
    echo "  - exp3_timecag_v3.py"
    echo "  - exp4_ablation_dmodel.py"
    echo "  - exp5_ablation_elayers.py"
    echo "  - exp6_ablation_dropout.py"
    echo "  - exp7_best_config.py"
    exit 1
fi

echo "✅ All experiment files found!"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel (or run with < /dev/null to skip)..." || echo "Auto-starting..."
echo ""

# Create results directory
mkdir -p $EXPERIMENTS_DIR/results

# Run experiments
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 1: Time-CAG v1 (seq_len=3000)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python $EXPERIMENTS_DIR/exp1_timecag_v1.py 2>&1 | tee $EXPERIMENTS_DIR/results/exp1_log.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 2: Time-CAG v2 (seq_len=1440)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python $EXPERIMENTS_DIR/exp2_timecag_v2.py 2>&1 | tee $EXPERIMENTS_DIR/results/exp2_log.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 3: Time-CAG v3 (seq_len=720, same as RAFT)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python $EXPERIMENTS_DIR/exp3_timecag_v3.py 2>&1 | tee $EXPERIMENTS_DIR/results/exp3_log.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 4: Ablation - Model Dimension (d_model)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python $EXPERIMENTS_DIR/exp4_ablation_dmodel.py 2>&1 | tee $EXPERIMENTS_DIR/results/exp4_log.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 5: Ablation - Encoder Layers (e_layers)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python $EXPERIMENTS_DIR/exp5_ablation_elayers.py 2>&1 | tee $EXPERIMENTS_DIR/results/exp5_log.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 6: Ablation - Dropout Rate"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python $EXPERIMENTS_DIR/exp6_ablation_dropout.py 2>&1 | tee $EXPERIMENTS_DIR/results/exp6_log.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 7: Best Configuration (Final)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python $EXPERIMENTS_DIR/exp7_best_config.py 2>&1 | tee $EXPERIMENTS_DIR/results/exp7_log.txt

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "ALL EXPERIMENTS COMPLETE!"
echo "════════════════════════════════════════════════════════════════"
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - $EXPERIMENTS_DIR/results/exp{1-7}_results.json"
echo "  - $EXPERIMENTS_DIR/results/exp{1-7}_log.txt"
echo ""

# Auto-download results (Colab only)
if [ -d "/content/RAFT" ]; then
    echo "📦 Creating results archive for download..."
    cd /content/RAFT
    zip -r results.zip experiments/results/
    
    echo "🚀 Triggering automatic download..."
    python3 << EOF
from google.colab import files
import os
if os.path.exists('/content/RAFT/results.zip'):
    print('Downloading results.zip...')
    files.download('/content/RAFT/results.zip')
    print('✅ Download started!')
else:
    print('❌ results.zip not found')
EOF
else
    echo "💻 Local run - results saved locally"
    echo "Next steps:"
    echo "  1. Update graphs with: python update_graphs.py"
    echo "  2. Check: graph{1-6}_*.png"
fi
echo ""
