#!/bin/bash
# Script to download pretrained models for EDSR and RCAN

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/data/models"
DATA_DIR="$PROJECT_DIR/data"

echo "Creating directories..."
mkdir -p "$MODELS_DIR"
mkdir -p "$DATA_DIR/test_images"

echo "=========================================="
echo "Downloading pretrained models..."
echo "=========================================="

# Note: This script provides instructions for downloading models
# Actual download URLs may vary. Users should:
# 1. Download EDSR models from: https://github.com/thstkdgus35/EDSR-PyTorch
# 2. Download RCAN models from: https://github.com/yulunzhang/RCAN
# 3. Convert PyTorch models to TensorFlow format if needed

echo ""
echo "EDSR Model:"
echo "  - Repository: https://github.com/thstkdgus35/EDSR-PyTorch"
echo "  - Pre-trained models available in the repository"
echo "  - For TensorFlow, you may need to convert from PyTorch format"
echo "  - Save to: $MODELS_DIR/edsr/"

echo ""
echo "RCAN Model:"
echo "  - Repository: https://github.com/yulunzhang/RCAN"
echo "  - Pre-trained models available in the repository"
echo "  - For TensorFlow, you may need to convert from PyTorch format"
echo "  - Save to: $MODELS_DIR/rcan/"

echo ""
echo "=========================================="
echo "Download script completed."
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download models from the repositories above"
echo "2. Convert models to TensorFlow format if needed"
echo "3. Place converted models in $MODELS_DIR/"
echo "4. Run test_local.py scripts to verify model loading"
