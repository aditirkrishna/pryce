# Autodiff Greeks Example — DerivLab

This example demonstrates:
- Monte Carlo simulation of a European call option using the DerivLab MLIR dialect
- Automatic differentiation (autodiff) to compute Greeks (Delta, Gamma)
- Analytical comparison with Black-Scholes formula
- Full pipeline: MLIR lowering, autodiff, LLVM IR, native code execution
- Data export and reproducibility

---

## How it Works

1. **`call_with_greeks.mlir`** — DerivLab input for pricing; autodiff is run to produce gradient function(s).
2. **`run_pipeline.sh`** — Lowers, autodiffs, compiles, and runs the simulation, producing `output.csv`.
3. **`black_scholes_with_greeks.py`** — Computes analytical price, Delta, and Gamma for comparison.
4. **`expected_lowered.mlir`** — Shows what the MLIR looks like after lowering and autodiff.
5. **`config.yaml`** — Input parameters for reproducibility.
6. **`output.csv`** — Output of MC simulation (price, Delta, Gamma, etc.).

---

## Run the Simulation

```bash
bash run_pipeline.sh
```

## Analytical Comparison

```bash
python3 black_scholes_with_greeks.py
```

## Output

- Monte Carlo price and Greeks: in `output.csv`
- Analytical price and Greeks: printed by Python script
- Compare the two for accuracy and convergence

---

## Folder Contents

| File                       | Purpose                                               |
|----------------------------|-------------------------------------------------------|
| call_with_greeks.mlir      | DerivLab input for MC pricing and autodiff            |
| expected_lowered.mlir      | Reference: Lowered MLIR (with autodiff)               |
| output.csv                 | MC simulation output (price, Delta, Gamma, etc.)      |
| black_scholes_with_greeks.py | Analytical price and Greeks (Python)                |
| run_pipeline.sh            | Full pipeline: lower, autodiff, compile, run          |
| config.yaml                | Input parameters for reproducibility                  |
| README.md                  | This documentation                                    |

---

This example demonstrates symbolic autodiff for risk sensitivities, compiler IR transformation, and advanced quant engineering.
