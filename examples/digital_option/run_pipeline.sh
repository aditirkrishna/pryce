#!/bin/bash
set -e

mlir-opt digital_option.mlir --lower-derivlab | \
mlir-opt --convert-math-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > lowered.mlir

mlir-translate lowered.mlir --mlir-to-llvmir > output.ll
llc output.ll -o output.s
clang output.s -o run_digital -lm
./run_digital > result.csv

echo "Digital option MC simulation output written to result.csv"
