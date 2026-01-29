#!/bin/bash
#
# Master Script: Run All Corrected PatchTST Experiments
# =======================================================
#
# This script runs the corrected PatchTST experiments with proper hyperparameters
# from the original ICLR 2023 paper.
#
# TOTAL RUNTIME: ~20-24 hours on single V100 GPU
#
# Usage:
#   bash run_all_corrected.sh [phase]
#
# Examples:
#   bash run_all_corrected.sh              # Run all phases sequentially
#   bash run_all_corrected.sh 1            # Run only Phase 1 (validation)
#   bash run_all_corrected.sh 2            # Run only Phase 2 (720 + 3000)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RAFT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}CORRECTED PATCHTST EXPERIMENTS - Master Run Script${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo "RAFT Root: $RAFT_ROOT"
echo "Results will be saved to: $SCRIPT_DIR/results/"
echo ""
echo -e "${YELLOW}⚠️  WARNING: This will take 20-24 hours on a single GPU!${NC}"
echo ""

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ ERROR: nvidia-smi not found. GPU required!${NC}"
    exit 1
fi

echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Confirm
read -p "Continue with experiment execution? (yes/no): " -r
echo
if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    echo "Aborted."
    exit 0
fi

# Determine which phases to run
PHASE=${1:-all}

# Create results directory
mkdir -p "$SCRIPT_DIR/results"
mkdir -p "$SCRIPT_DIR/logs"

# Log file
LOGFILE="$SCRIPT_DIR/logs/run_all_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOGFILE"
echo ""

# Function to run experiment
run_experiment() {
    local phase=$1
    local name=$2
    local script=$3
    local expected_hours=$4
    
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${GREEN}▶ Running: $name${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
    echo "Script: $script"
    echo "Expected runtime: ~$expected_hours hours"
    echo "Started: $(date)"
    echo ""
    
    START_TIME=$(date +%s)
    
    # Run experiment
    cd "$RAFT_ROOT"
    python3 "$script" 2>&1 | tee -a "$LOGFILE"
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINS=$(((ELAPSED % 3600) / 60))
    
    echo ""
    echo -e "${GREEN}✅ Completed: $name${NC}"
    echo "Actual runtime: ${HOURS}h ${MINS}m"
    echo "Completed: $(date)"
    echo ""
    
    # Check if results file exists
    RESULT_FILE="$SCRIPT_DIR/results/${phase}*.json"
    if ls $RESULT_FILE 1> /dev/null 2>&1; then
        echo -e "${GREEN}✅ Results saved successfully${NC}"
        echo ""
    else
        echo -e "${RED}⚠️  WARNING: Results file not found!${NC}"
        echo ""
    fi
}

# ================================================================================================
# PHASE 1: VALIDATION
# ================================================================================================
if [[ "$PHASE" == "all" ]] || [[ "$PHASE" == "1" ]]; then
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                              PHASE 1: VALIDATION                                             ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Goal: Reproduce original PatchTST paper results (seq_len=336)"
    echo "Expected MSE: 0.377-0.380"
    echo ""
    echo -e "${YELLOW}⚠️  CRITICAL: If this fails, implementation is broken - do not proceed!${NC}"
    echo ""
    
    run_experiment \
        "phase1" \
        "Validation (PatchTST-336 Original)" \
        "experiments_patchtst/corrected/phase1_patchtst_336_validation.py" \
        "4"
    
    # Check validation status
    VALIDATION_FILE="$SCRIPT_DIR/results/phase1_validation_336.json"
    if [ -f "$VALIDATION_FILE" ]; then
        STATUS=$(python3 -c "import json; print(json.load(open('$VALIDATION_FILE'))['validation_status'])")
        
        if [[ "$STATUS" == "FAILED" ]]; then
            echo -e "${RED}╔══════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${RED}║                           ❌ VALIDATION FAILED!                                              ║${NC}"
            echo -e "${RED}╚══════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            echo -e "${RED}CRITICAL ERROR: Implementation does not match original paper!${NC}"
            echo ""
            echo "DO NOT PROCEED to Phase 2!"
            echo ""
            echo "Next steps:"
            echo "  1. Review error logs in $LOGFILE"
            echo "  2. Check model implementation in models/PatchTST.py"
            echo "  3. Verify data loading"
            echo "  4. Compare with original PatchTST repository"
            echo ""
            exit 1
        else
            echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${GREEN}║                           ✅ VALIDATION PASSED!                                              ║${NC}"
            echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            echo "Implementation is correct! Safe to proceed to Phase 2."
            echo ""
        fi
    else
        echo -e "${RED}⚠️  WARNING: Validation results file not found!${NC}"
        echo "Cannot verify validation status - proceed with caution"
        echo ""
    fi
fi

# ================================================================================================
# PHASE 2: CRITICAL TEST (720 and 3000)
# ================================================================================================
if [[ "$PHASE" == "all" ]] || [[ "$PHASE" == "2" ]]; then
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                         PHASE 2: CRITICAL COMPARISON TEST                                    ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Goal: Test if PatchTST degrades with longer context (corrected configs)"
    echo ""
    echo "This phase determines the validity of our paper!"
    echo ""
    
    # Phase 2A: PatchTST-720
    run_experiment \
        "phase2a" \
        "PatchTST-720 (Corrected)" \
        "experiments_patchtst/corrected/phase2a_patchtst_720_corrected.py" \
        "3-4"
    
    # Phase 2B: PatchTST-3000 (THE CRITICAL TEST)
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                      🚨 CRITICAL TEST - PatchTST-3000                                        ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "This experiment determines our paper scenario:"
    echo "  • MSE < 0.45  → Original paper correct (major revision needed)"
    echo "  • MSE > 0.50  → Our finding validated (proceed with paper)"
    echo ""
    
    run_experiment \
        "phase2b" \
        "PatchTST-3000 (Corrected) - CRITICAL" \
        "experiments_patchtst/corrected/phase2b_patchtst_3000_corrected_CRITICAL.py" \
        "8-12"
    
    # Analyze Phase 2 results
    PHASE2B_FILE="$SCRIPT_DIR/results/phase2b_patchtst_3000_corrected_CRITICAL.json"
    if [ -f "$PHASE2B_FILE" ]; then
        echo ""
        echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BLUE}║                           PHASE 2 RESULTS SUMMARY                                            ║${NC}"
        echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        SCENARIO=$(python3 -c "import json; print(json.load(open('$PHASE2B_FILE'))['scenario'])")
        RECOMMENDATION=$(python3 -c "import json; print(json.load(open('$PHASE2B_FILE'))['recommendation'])")
        MSE_3000=$(python3 -c "import json; print(json.load(open('$PHASE2B_FILE'))['test_mse'])")
        
        echo "Scenario: $SCENARIO"
        echo "PatchTST-3000 MSE: $MSE_3000"
        echo "Recommendation: $RECOMMENDATION"
        echo ""
        
        if [[ "$SCENARIO" == "OUR_FINDING_VALIDATED" ]]; then
            echo -e "${GREEN}✅ SUCCESS: Our finding is validated!${NC}"
            echo ""
            echo "Next steps:"
            echo "  1. Run Phase 3 (cross-domain) to strengthen findings"
            echo "  2. Prepare paper for submission"
            echo ""
        elif [[ "$SCENARIO" == "ORIGINAL_PAPER_CORRECT" ]]; then
            echo -e "${RED}⚠️  ATTENTION: Original paper is correct${NC}"
            echo ""
            echo "Next steps:"
            echo "  1. Begin major paper revision"
            echo "  2. Reframe contribution"
            echo "  3. Acknowledge configuration issues"
            echo ""
        else
            echo -e "${YELLOW}⚠️  AMBIGUOUS: Results unclear${NC}"
            echo ""
            echo "Next steps:"
            echo "  1. Run Phase 3 for additional data"
            echo "  2. Analyze training curves"
            echo "  3. Consider ensemble runs"
            echo ""
        fi
    fi
fi

# ================================================================================================
# COMPLETION
# ================================================================================================
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                          ALL EXPERIMENTS COMPLETED                                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Results directory: $SCRIPT_DIR/results/"
echo "Full log: $LOGFILE"
echo ""
echo "Completed: $(date)"
echo ""

# Generate summary report
SUMMARY_FILE="$SCRIPT_DIR/results/EXPERIMENT_SUMMARY.txt"
echo "Generating summary report..."
echo ""
echo "================================================================================================" > "$SUMMARY_FILE"
echo "CORRECTED PATCHTST EXPERIMENTS - SUMMARY" >> "$SUMMARY_FILE"
echo "================================================================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Add results from all phases
for result_file in "$SCRIPT_DIR/results"/*.json; do
    if [ -f "$result_file" ]; then
        echo "----------------------------------------" >> "$SUMMARY_FILE"
        echo "File: $(basename "$result_file")" >> "$SUMMARY_FILE"
        echo "----------------------------------------" >> "$SUMMARY_FILE"
        python3 -c "import json; import sys; data = json.load(open('$result_file')); print('Experiment:', data.get('experiment_name', 'N/A')); print('MSE:', data.get('test_mse', 'N/A')); print('MAE:', data.get('test_mae', 'N/A')); print('Status:', data.get('status', data.get('validation_status', data.get('scenario', 'N/A'))))" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    fi
done

echo "Summary report saved to: $SUMMARY_FILE"
echo ""
echo -e "${GREEN}✅ ALL DONE!${NC}"
echo ""
