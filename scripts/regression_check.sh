#!/bin/bash
# Regression test: compare current output to golden outputs
set -e

MLIR_OPT=mlir-opt
MLIR_TRANSLATE=mlir-translate

TEST_MLIR="$(dirname "$0")/../test/unit/test_simulate_gbm.mlir"
GOLDEN_LL="$(dirname "$0")/../test/golden/test_simulate_gbm.ll"
TMP_LL="sim_gbm_current.ll"

# Lower DerivLab to LLVM dialects
$MLIR_OPT --lower-derivlab "$TEST_MLIR" -o "$TMP_LL"

# Optionally, translate to LLVM IR text (if needed)
$MLIR_TRANSLATE --mlir-to-llvmir "$TMP_LL" > "$TMP_LL.ir"

# Compare current output to golden
if diff -u "$TMP_LL.ir" "$GOLDEN_LL"; then
  echo "[PASS] Output matches golden file."
else
  echo "[FAIL] Output differs from golden file!"
  exit 1
fi
