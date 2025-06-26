#!/bin/bash
set -e

echo "Running Antithetic Variates Example..."
mlir-opt call_antithetic.mlir | mlir-translate --mlir-to-llvmir | llc | clang -x assembler - -o call_antithetic.exe -lm
./call_antithetic.exe > output_antithetic.txt

echo "Running Control Variate Example..."
mlir-opt call_control_variate.mlir | mlir-translate --mlir-to-llvmir | llc | clang -x assembler - -o call_control_variate.exe -lm
./call_control_variate.exe > output_control_variate.txt

echo "Running Black-Scholes Analytical..."
python3 bs_call.py > output_bs.txt

echo "All variance reduction examples complete. See output_*.txt for results."
