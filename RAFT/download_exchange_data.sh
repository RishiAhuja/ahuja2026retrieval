#!/bin/bash
# Download Exchange Rate dataset from Time Series Library

echo "================================"
echo "Exchange Rate Dataset Download"
echo "================================"
echo ""

# Create directory
DATASET_DIR="/Users/rishi/StudioProjects/caft/RAFT/data/exchange_rate"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "📁 Target directory: $DATASET_DIR"
echo ""

# Download from Time Series Library GitHub
echo "📥 Downloading exchange_rate.csv from Time Series Library..."
echo ""

# Option 1: Direct download (if available)
curl -L -o exchange_rate.csv \
  "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/exchange_rate/exchange_rate.csv"

# Check if download succeeded
if [ -f "exchange_rate.csv" ]; then
    FILE_SIZE=$(wc -c < exchange_rate.csv)
    if [ "$FILE_SIZE" -gt 1000 ]; then
        echo "✅ Download successful!"
        echo "   File size: $FILE_SIZE bytes"
        echo ""
        echo "📊 Dataset info:"
        head -n 3 exchange_rate.csv
        echo ""
        echo "✅ Ready to run experiments!"
    else
        echo "⚠️  Downloaded file seems too small. Trying alternative method..."
        rm exchange_rate.csv
        
        echo ""
        echo "📖 Manual download instructions:"
        echo "   1. Go to: https://github.com/thuml/Time-Series-Library"
        echo "   2. Navigate to: dataset/exchange_rate/"
        echo "   3. Download: exchange_rate.csv"
        echo "   4. Place in: $DATASET_DIR"
    fi
else
    echo "❌ Automatic download failed."
    echo ""
    echo "📖 Manual download instructions:"
    echo ""
    echo "Option A: Clone entire Time Series Library"
    echo "   cd /Users/rishi/StudioProjects/caft/RAFT/data"
    echo "   git clone https://github.com/thuml/Time-Series-Library.git"
    echo "   cp Time-Series-Library/dataset/exchange_rate/exchange_rate.csv exchange_rate/"
    echo ""
    echo "Option B: Download from Hugging Face datasets"
    echo "   pip install datasets"
    echo "   python -c \"from datasets import load_dataset; ds = load_dataset('thuml/time_series_benchmark', 'exchange_rate'); ...\""
    echo ""
    echo "Option C: Download from original source"
    echo "   Visit: https://github.com/laiguokun/multivariate-time-series-data"
    echo "   Dataset: exchange_rate.txt (8 exchange rates)"
    echo ""
fi

echo ""
echo "================================"
