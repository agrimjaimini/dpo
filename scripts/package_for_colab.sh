#!/bin/bash
# Package DPO project for Google Colab upload

echo "================================"
echo "Packaging DPO for Google Colab"
echo "================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR/.."

# Create zip excluding unnecessary files
echo ""
echo "Creating dpo.zip..."
zip -r dpo.zip dpo/ \
    -x "dpo/venv/*" \
    -x "dpo/.git/*" \
    -x "dpo/.venv/*" \
    -x "dpo/outputs/*" \
    -x "dpo/__pycache__/*" \
    -x "dpo/*/__pycache__/*" \
    -x "dpo/*/*/__pycache__/*" \
    -x "dpo/.DS_Store" \
    -x "dpo/*/.DS_Store" \
    -x "dpo/*.pyc" \
    -x "dpo/.ipynb_checkpoints/*"

# Get size
SIZE=$(du -h dpo.zip | cut -f1)

echo ""
echo "✓ Package created: dpo.zip ($SIZE)"
echo ""
echo "Next steps:"
echo "1. Go to https://colab.research.google.com/"
echo "2. Upload DPO_Training_Colab.ipynb"
echo "3. Change runtime to GPU (Runtime → Change runtime type → GPU)"
echo "4. Run the cells and upload dpo.zip when prompted"
echo ""
echo "Happy training! 🚀"
