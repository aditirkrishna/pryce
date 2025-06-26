#!/bin/bash
set -e

# Lower DerivLab to Standard/LLVM dialects
mlir-opt call_option.mlir --lower-derivlab | \
mlir-opt --convert-math-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > lowered.mlir

# Translate to LLVM IR
mlir-translate lowered.mlir --mlir-to-llvmir > output.ll

# Compile to native assembly
llc output.ll -o output.s
clang output.s -o derivlab_exec -lm

# Run and capture output
./derivlab_exec > output.csv

echo "Monte Carlo simulation output written to output.csv"
