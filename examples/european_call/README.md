# European Call Option Pricing — DerivLab Example

This advanced example demonstrates:
- Monte Carlo simulation of a European call option using the DerivLab MLIR dialect
- Analytical price comparison using the Black-Scholes formula
- Full pipeline: MLIR lowering, LLVM IR, native code execution
- Data export and reproducibility

---

## How it Works

1. **`call_option.mlir`** — Main input in the DerivLab dialect. Simulates GBM, computes call payoff, averages, discounts.
2. **`run_pipeline.sh`** — Lowers, compiles, and runs the simulation, producing `output.csv`.
3. **`black_scholes.py`** — Computes the analytical price for the same parameters.
4. **`expected_lowered.mlir`** — Shows what the MLIR looks like after lowering to standard/LLVM dialects.
5. **`config.yaml`** — Input parameters for easy adjustment and reproducibility.
6. **`output.csv`** — Output of the MC simulation (price, variance, paths, etc.).

---

## Run the Simulation

```bash
bash run_pipeline.sh
```

## Analytical Comparison

```bash
python3 black_scholes.py
```

## Output

- Monte Carlo price: in `output.csv`
- Analytical price: printed by Python script
- Compare the two for accuracy and convergence

---

## Folder Contents

| File                  | Purpose                                          |
|-----------------------|--------------------------------------------------|
| call_option.mlir      | DerivLab dialect input for MC simulation         |
| expected_lowered.mlir | Reference: Lowered MLIR (standard/LLVM dialects) |
| output.csv            | MC simulation output (price, etc.)               |
| black_scholes.py      | Analytical Black-Scholes price (Python)          |
| run_pipeline.sh       | Full pipeline: lower, compile, run               |
| config.yaml           | Input parameters for reproducibility              |
| README.md             | This documentation                               |

---

This example is designed for clarity, reproducibility, and as a showcase for advanced MLIR/quant engineering.
