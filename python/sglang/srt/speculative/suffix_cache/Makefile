# Makefile for suffix_cache C++ extension
# Simple alternative to build.sh for users who prefer make

.PHONY: all build clean test install help

PYTHON ?= python
PIP ?= pip

all: build

help:
	@echo "Available targets:"
	@echo "  make build    - Build the C++ extension (default)"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make test     - Test the built extension"
	@echo "  make install  - Install in development mode"
	@echo "  make deps     - Install build dependencies"
	@echo ""
	@echo "Environment variables:"
	@echo "  PYTHON        - Python executable (default: python)"
	@echo "  PIP           - Pip executable (default: pip)"

deps:
	@echo "Installing build dependencies..."
	$(PIP) install pybind11 ninja torch

build:
	@echo "Building suffix_cache C++ extension..."
	$(PYTHON) setup.py build_ext --inplace
	@echo ""
	@echo "Build complete! Generated files:"
	@ls -lh _C.cpython-*.so 2>/dev/null || echo "No .so files found"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build dist *.egg-info
	rm -f _C.cpython-*.so _C.*.pyd
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete!"

test:
	@echo "Testing suffix_cache extension..."
	$(PYTHON) -c "from sglang.srt.speculative.suffix_cache._C import SuffixTree, Candidate; print('✓ Import successful!')"
	@echo ""
	@echo "Running basic functionality test..."
	$(PYTHON) -c "\
from sglang.srt.speculative.suffix_cache._C import SuffixTree; \
tree = SuffixTree(64); \
tree.extend(0, [1, 2, 3, 4, 5]); \
candidate = tree.speculate([3, 4], 8, 1.0, 0.0, 0.1, False); \
print(f'✓ Speculation test passed: {len(candidate.token_ids)} tokens')"

install: deps build
	@echo "Installing in development mode..."
	$(PIP) install -e .
	@echo "Installation complete!"

rebuild: clean build
	@echo "Rebuild complete!"
