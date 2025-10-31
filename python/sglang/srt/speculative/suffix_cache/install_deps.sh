#!/bin/bash
# Install/upgrade dependencies for suffix_cache compilation
# This script ensures all dependencies are at the correct versions for Python 3.12

set -e

echo "=================================================="
echo "Installing dependencies for suffix_cache"
echo "=================================================="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

echo "Step 1: Installing/upgrading pybind11..."
echo "  (Need pybind11 >= 2.10.0 for Python 3.12 support)"
pip install --upgrade "pybind11>=2.10.0"
echo "  Installed version: $(python -c 'import pybind11; print(pybind11.__version__)')"
echo ""

echo "Step 2: Installing ninja build system (optional, for faster builds)..."
pip install ninja || echo "  Warning: ninja installation failed, will use default build system"
echo ""

echo "Step 3: Checking PyTorch..."
if python -c "import torch" 2>/dev/null; then
    echo "  ✓ PyTorch is installed: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "  ✗ PyTorch is not installed!"
    echo "  Please install PyTorch first: https://pytorch.org/get-started/locally/"
    exit 1
fi
echo ""

echo "Step 4: Installing packaging module (for version checks)..."
pip install packaging
echo ""

echo "=================================================="
echo "✓ All dependencies installed successfully!"
echo "=================================================="
echo ""
echo "Now you can run: ./build.sh"
