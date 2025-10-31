#!/bin/bash
# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build script for suffix_cache C++ extension
# This script compiles the C++ code and places the .so file in the correct location

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "Building suffix_cache C++ extension"
echo "=================================================="
echo "Working directory: $(pwd)"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check pybind11 version - need 2.10+ for Python 3.12
echo "Checking pybind11 version..."
PYBIND11_VERSION=$(python -c "import pybind11; print(pybind11.__version__)" 2>/dev/null || echo "0.0.0")
REQUIRED_VERSION="2.10.0"

if ! python -c "import pybind11" 2>/dev/null; then
    echo "pybind11 is not installed. Installing latest version..."
    pip install "pybind11>=2.10.0"
elif python -c "from packaging import version; import pybind11; exit(0 if version.parse(pybind11.__version__) >= version.parse('2.10.0') else 1)" 2>/dev/null; then
    echo "✓ pybind11 version $PYBIND11_VERSION is compatible with Python 3.12"
else
    echo "WARNING: pybind11 version $PYBIND11_VERSION is too old for Python 3.12"
    echo "Upgrading pybind11 to latest version..."
    pip install --upgrade "pybind11>=2.10.0"
fi

# Check if ninja is installed (optional, but speeds up compilation)
if ! python -c "import ninja" 2>/dev/null; then
    echo "WARNING: ninja build system not found. Installing for faster compilation..."
    pip install ninja
fi

# Check if torch is installed
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch is not installed. Please install PyTorch first."
    exit 1
fi

echo "Building C++ extension..."
echo ""

# Clean previous build artifacts
if [ -d "build" ]; then
    echo "Cleaning previous build artifacts..."
    rm -rf build
fi

# Build the extension
# Use pip install -e . to build in development mode
# This will compile and place the .so file in the correct location
python setup.py build_ext --inplace

echo ""
echo "=================================================="
echo "Build completed successfully!"
echo "=================================================="

# Find and display the generated .so file
SO_FILE=$(find . -name "_C.cpython-*.so" | head -1)
if [ -n "$SO_FILE" ]; then
    echo "Generated file: $SO_FILE"
    ls -lh "$SO_FILE"

    # Verify it's in the correct location
    if [[ "$SO_FILE" == "./_C.cpython-"* ]]; then
        echo "✓ File is in the correct location: $(pwd)"
    else
        echo "WARNING: File might not be in the expected location"
    fi
else
    echo "WARNING: Could not find generated .so file"
    echo "Searching for all .so files:"
    find . -name "*.so"
fi

echo ""
echo "To test the extension, run:"
echo "  python -c 'from sglang.srt.speculative.suffix_cache._C import SuffixTree; print(\"Import successful!\")'"
