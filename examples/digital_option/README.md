# Digital (Binary) Call Option Pricing — DerivLab Example

This example demonstrates:
- Monte Carlo simulation of a digital (binary) call option using the DerivLab MLIR dialect
- Analytical price comparison using the Black-Scholes digital formula
- Full pipeline: MLIR lowering, LLVM IR, native code execution
- Data export and reproducibility

---

## How it Works

1. **`digital_option.mlir`** — Main input in the DerivLab dialect. Simulates GBM, computes digital call payoff, averages, discounts.
2. **`run_pipeline.sh`** — Lowers, compiles, and runs the simulation, producing `result.csv`.
3. **`black_scholes_digital.py`** — Computes the analytical digital price for the same parameters.
4. **`expected_lowered.mlir`** — Shows what the MLIR looks like after lowering to standard/LLVM dialects.
5. **`config.yaml`** — Input parameters for easy adjustment and reproducibility.
6. **`result.csv`** — Output of the MC simulation (price, variance, paths, etc.).

---

## Run the Simulation

```bash
bash run_pipeline.sh
```

## Analytical Comparison

```bash
python3 black_scholes_digital.py
```

## Output

- Monte Carlo price: in `result.csv`
- Analytical price: printed by Python script
- Compare the two for accuracy and convergence

---

## Folder Contents

| File                    | Purpose                                           |
|-------------------------|---------------------------------------------------|
| digital_option.mlir     | DerivLab dialect input for MC simulation          |
| expected_lowered.mlir   | Reference: Lowered MLIR (standard/LLVM dialects)  |
| result.csv              | MC simulation output (price, etc.)                |
| black_scholes_digital.py| Analytical digital call price (Python)            |
| run_pipeline.sh         | Full pipeline: lower, compile, run                |
| config.yaml             | Input parameters for reproducibility              |
| README.md               | This documentation                                |

---

This example demonstrates non-linear payoffs, conditional logic in MLIR lowering, and the extensibility of your DerivLab dialect.
