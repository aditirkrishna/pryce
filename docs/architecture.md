# DerivLab Compiler Architecture

DerivLab is an MLIR-based compiler infrastructure for quantitative finance and derivatives modeling. Its pipeline is designed to:

- Parse and represent finance-specific operations as custom MLIR dialects
- Lower these operations to standard MLIR and LLVM dialects
- Generate efficient native code for simulation and option pricing

## Pipeline Overview

1. **Frontend**: Parses DerivLab syntax and emits MLIR with custom `derivlab` dialect ops (e.g., `simulate_gbm`, `payoff`).
2. **Dialect Lowering**: Custom passes lower `derivlab` ops to standard MLIR/LLVM dialects using C++ logic (see `lib/Transforms`).
3. **Code Generation**: MLIR is further lowered to LLVM IR and compiled to native code.
4. **Runtime**: Optional C++ runtime helpers provide utilities for simulation and option payoff calculations.

## Main Components
- `mlir/include/DerivLab/DerivLabOps.td`: TableGen definitions for custom operations
- `mlir/lib/Dialect/DerivLabDialect.cpp`: C++ dialect registration and op hooks
- `mlir/lib/Transforms/`: Lowering logic for custom ops
- `runtime/`: C++ runtime helpers (e.g., Black-Scholes)
- `test/`: MLIR test cases and examples

See also the pipeline diagram (`docs/derivlab_pipeline.svg`).
