#!/bin/bash
# Copy PatchTST dependencies from Time-Series-Library

echo "Copying PatchTST layer dependencies..."

# Clone TSL if not exists
if [ ! -d "temp_tsl" ]; then
    echo "Cloning Time-Series-Library..."
    git clone https://github.com/thuml/Time-Series-Library.git temp_tsl
fi

# Copy required layer files
echo "Copying layer files..."
cp temp_tsl/layers/Transformer_EncDec.py ./layers/
cp temp_tsl/layers/SelfAttention_Family.py ./layers/
cp temp_tsl/layers/Embed.py ./layers/

echo "✅ Layer dependencies copied"

# Cleanup
rm -rf temp_tsl
echo "✅ Cleanup complete"

echo ""
echo "Ready to run PatchTST experiments!"
