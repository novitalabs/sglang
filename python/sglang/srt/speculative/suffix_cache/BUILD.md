# Building suffix_cache C++ Extension

This document describes how to compile the suffix_cache C++ extension.

## Quick Start

The simplest way to build the extension:

```bash
cd /path/to/sglang/python/sglang/srt/speculative/suffix_cache
./build.sh
```

## Prerequisites

Before building, ensure you have:

1. **Python 3.12** (or compatible version)
2. **PyTorch** installed
3. **pybind11** (will be auto-installed by build script if missing)
4. **CMake** >= 3.14
5. **C++ compiler** with C++17 support (g++, clang, or MSVC)
6. **ninja** (optional, for faster builds - will be auto-installed if missing)

Install prerequisites:
```bash
pip install torch pybind11 ninja
```

## Build Methods

### Method 1: Using build.sh (Recommended)

```bash
./build.sh
```

This script will:
- Check dependencies
- Clean previous build artifacts
- Compile the C++ extension
- Place `_C.cpython-312-x86_64-linux-gnu.so` in the current directory

### Method 2: Using setup.py directly

```bash
python setup.py build_ext --inplace
```

This uses CMake to compile the extension and places the `.so` file in the current directory.

### Method 3: Installing in development mode

```bash
pip install -e .
```

This builds and installs the extension in development mode, allowing you to modify the code without reinstalling.

## Output

After successful compilation, you should see:
```
_C.cpython-312-x86_64-linux-gnu.so
```

The filename may vary based on your Python version and platform:
- **Linux**: `_C.cpython-312-x86_64-linux-gnu.so`
- **macOS**: `_C.cpython-312-darwin.so`
- **Windows**: `_C.cp312-win_amd64.pyd`

## Verification

Test that the extension was built correctly:

```bash
python -c "from sglang.srt.speculative.suffix_cache._C import SuffixTree; print('Import successful!')"
```

Or test the full functionality:

```python
from sglang.srt.speculative.suffix_cache._C import SuffixTree, Candidate

# Create a suffix tree with max depth of 64
tree = SuffixTree(64)

# Add some tokens to sequence 0
tree.extend(0, [1, 2, 3, 4, 5])

# Speculate based on pattern [3, 4]
candidate = tree.speculate(
    pattern=[3, 4],
    max_spec_tokens=8,
    max_spec_factor=1.0,
    max_spec_offset=0.0,
    min_token_prob=0.1,
    use_tree_spec=False
)

print(f"Speculated tokens: {candidate.token_ids}")
print(f"Match length: {candidate.match_len}")
print(f"Score: {candidate.score}")
```

## Troubleshooting

### CMake not found
```bash
pip install cmake
# or on Ubuntu/Debian:
sudo apt-get install cmake
```

### pybind11 not found
```bash
pip install pybind11
```

### Compilation errors with C++17
Ensure your compiler supports C++17:
- GCC >= 7.0
- Clang >= 5.0
- MSVC >= 19.14 (Visual Studio 2017 15.7)

### Wrong Python version in .so filename
The script uses the currently active Python. Ensure you're in the correct virtual environment:
```bash
which python
python --version
```

### Build artifacts causing issues
Clean build artifacts:
```bash
rm -rf build *.so
./build.sh
```

## Manual Build (Advanced)

If you need fine-grained control:

```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ../csrc/suffix_cache \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DTORCH_CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

# Compile
cmake --build . -j$(nproc)

# Copy .so file back to parent directory
cp _C.cpython-*.so ..
```

## Build System Overview

The build system uses:
- **setup.py**: Python build orchestrator using setuptools
- **CMakeLists.txt**: CMake build configuration in `csrc/suffix_cache/`
- **pybind11**: C++/Python binding library
- Source files:
  - `csrc/suffix_cache/pybind.cc`: Python bindings
  - `csrc/suffix_cache/suffix_tree.cc`: Core implementation
  - `csrc/suffix_cache/suffix_tree.h`: Header file

## CI/CD Integration

For automated builds:

```bash
# Install dependencies
pip install pybind11 ninja torch

# Build
cd python/sglang/srt/speculative/suffix_cache
python setup.py build_ext --inplace

# Test
python -c "from sglang.srt.speculative.suffix_cache._C import SuffixTree"
```
