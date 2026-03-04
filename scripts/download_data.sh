#!/bin/bash
# =============================================================================
# Download ETT datasets
# =============================================================================
# Downloads the Electricity Transformer Temperature (ETT) datasets
# used in the paper experiments.

set -e
mkdir -p data/ETT
cd data/ETT

BASE_URL="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"

for file in ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv; do
    if [ ! -f "$file" ]; then
        echo "Downloading $file..."
        curl -sL "${BASE_URL}/${file}" -o "$file"
        echo "  Done: $(wc -l < "$file") rows"
    else
        echo "$file already exists, skipping"
    fi
done

echo ""
echo "All ETT datasets downloaded to data/ETT/"
