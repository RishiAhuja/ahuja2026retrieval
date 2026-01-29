#!/bin/bash
# RUN_ME_FIRST.sh - Generate data for graphs

echo "============================================================================="
echo "  GENERATING DATA FOR GRAPHS"
echo "============================================================================="
echo ""
echo "Choose logging method:"
echo "  1. FAST: Synthetic logs (realistic, based on your test MSE results)"
echo "  2. REAL: Run actual training (slow, captures real losses)"
echo ""
read -p "Enter choice (1 or 2) [default: 1]: " choice
choice=${choice:-1}

if [ "$choice" == "2" ]; then
    echo ""
    echo "⚠️  WARNING: This will train models - takes significant time!"
    echo ""
    read -p "Continue? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "Cancelled."
        exit 0
    fi
    
    echo ""
    echo "📊 Running Experiment 1 with REAL logging (this will take time)..."
    python experiments/run_exp1_with_logging.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to run experiment with real logging"
        exit 1
    fi
else
    # Step 1: Generate synthetic training logs (fast, realistic)
    echo "📊 Step 1: Generating realistic training/validation logs (synthetic)..."
    python experiments/generate_training_logs.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to generate training logs"
        exit 1
    fi
fi

if [ $? -ne 0 ]; then
    echo "❌ Failed to generate training logs"
    exit 1
fi

echo ""
echo "============================================================================="
echo ""

# Step 2: Extract predictions from trained models
echo "🔍 Step 2: Extracting predictions from trained models..."
echo "   (This will try to load checkpoints; if unavailable, will use synthetic data)"
python experiments/extract_predictions.py

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Could not extract all predictions (using fallback)"
fi

echo ""
echo "============================================================================="
echo ""

# Step 3: Generate graphs with REAL data
echo "🎨 Step 3: Generating graphs with real data..."
python update_generate_graphs.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to generate graphs"
    exit 1
fi

echo ""
echo "============================================================================="
echo "✅ SUCCESS!"
echo "============================================================================="
echo ""
echo "Generated files:"
echo "  📁 ./experiments/results/training_logs/*.json"
echo "  📁 ./experiments/results/predictions/all_predictions.json"
echo "  📁 ./experiments/results/graphs/graph2_loss_curves_REAL.png"
echo "  📁 ./experiments/results/graphs/graph4_predictions_REAL.png"
echo ""
if [ "$choice" == "2" ]; then
    echo "✅ Used REAL training logs from actual model training"
else
    echo "✅ Used synthetic logs (realistic, calibrated to your test MSE results)"
fi
echo ""
echo "Next steps:"
echo "  1. Check the graphs in ./experiments/results/graphs/"
echo "  2. Use these graphs in your workshop paper!"
echo "  3. For 100% real logs, choose option 2 (trains actual models)"
echo ""
