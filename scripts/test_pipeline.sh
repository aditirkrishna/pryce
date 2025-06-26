#!/bin/bash
# Integration test: run MLIR file through full pipeline and check output
set -e

MLIR_OPT=mlir-opt
MLIR_TRANSLATE=mlir-translate
LLC=llc
CLANG=clang

TEST_MLIR="$(dirname "$0")/../test/integration/test_pipeline.mlir"
OUTPUT_LL="test_pipeline.ll"
OUTPUT_S="test_pipeline.s"
OUTPUT_EXE="test_pipeline.exe"

# Lower DerivLab dialect to standard/LLVM dialects
$MLIR_OPT --lower-derivlab "$TEST_MLIR" -o "$OUTPUT_LL"

# Translate MLIR LLVM IR to textual LLVM IR (if needed)
$MLIR_TRANSLATE --mlir-to-llvmir "$OUTPUT_LL" > "$OUTPUT_LL.ir"

# Compile to assembly
$LLC "$OUTPUT_LL.ir" -o "$OUTPUT_S"

# Compile to executable
$CLANG "$OUTPUT_S" -o "$OUTPUT_EXE" -lm

# Run and check output (example: expecting a float value)
./"$OUTPUT_EXE"
