#!/bin/bash
# Setup environment for DerivLab

# Check for MLIR installation
if ! command -v mlir-opt &> /dev/null
then
    echo "MLIR tools not found. Please install MLIR and ensure mlir-opt is in your PATH."
    exit 1
fi

# Set up environment variables (example)
export DERIVLAB_ROOT=$(pwd)
export MLIR_PATH=$(dirname $(which mlir-opt))

echo "Environment setup complete."
