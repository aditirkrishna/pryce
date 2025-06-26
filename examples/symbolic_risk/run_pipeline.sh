#!/bin/bash
set -e

# Lower symbolic contract/risk ops to MC kernels and risk hooks
mlir-opt risk_test.mlir --lower-contracts --risk-autodiff | \
mlir-opt --convert-math-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > lowered.mlir

# Translate to LLVM IR
mlir-translate lowered.mlir --mlir-to-llvmir > output.ll
llc output.ll -o output.s
clang output.s -o run_risk -lm
./run_risk > output.csv

echo "Symbolic risk engine output written to output.csv"
