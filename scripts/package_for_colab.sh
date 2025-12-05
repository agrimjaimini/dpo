#!/bin/bash
# Package DPO project for Google Colab upload

echo "================================"
echo "Packaging DPO for Google Colab"
echo "================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Go to project directory and zip contents (not the folder itself)
cd "$PROJECT_DIR"

# Create zip excluding unnecessary files
echo ""
echo "Creating dpo.zip..."
zip -r ../dpo.zip . \
    -x "venv/*" \
    -x ".git/*" \
    -x ".venv/*" \
    -x "outputs/*" \
    -x "__pycache__/*" \
    -x "*/__pycache__/*" \
    -x "*/*/__pycache__/*" \
    -x ".DS_Store" \
    -x "*/.DS_Store" \
    -x "*.pyc" \
    -x ".ipynb_checkpoints/*" \
    -x ".*"

# Move to parent directory for easy access
cd ..

# Get size
SIZE=$(du -h dpo.zip | cut -f1)

echo ""
echo "✓ Package created: dpo.zip ($SIZE)"
echo "✓ Location: $(pwd)/dpo.zip"
echo ""
echo "Next steps:"
echo "1. Go to https://colab.research.google.com/"
echo "2. Upload DPO_Training_Colab.ipynb"
echo "3. Change runtime to GPU (Runtime → Change runtime type → GPU)"
echo "4. Run the cells and upload dpo.zip when prompted"
echo ""
echo "Happy training! 🚀"
