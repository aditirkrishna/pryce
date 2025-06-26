#!/bin/bash
set -e

# Lower to Standard/LLVM dialects
mlir-opt call_with_greeks.mlir --lower-derivlab | \
mlir-opt --convert-math-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > lowered.mlir

# Optionally run autodiff (Enzyme or custom pass)
enzyme-opt lowered.mlir --autodiff @price_call --wrt-arg=0 --ret-derivative > autodiff.mlir

# Translate to LLVM IR
mlir-translate autodiff.mlir --mlir-to-llvmir > output.ll
llc output.ll -o output.s
clang output.s -o run_greeks -lm
./run_greeks > output.csv

echo "Greeks MC simulation output written to output.csv"
