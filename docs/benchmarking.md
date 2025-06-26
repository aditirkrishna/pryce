# Benchmarking DerivLab Monte Carlo Pricing

This document describes how to benchmark the Monte Carlo (MC) pricing pipeline in the DerivLab MLIR project.

## What is Benchmarked?
- **MC pipeline execution time** for varying numbers of simulated paths.
- **Parameters:** spot, strike, rate, volatility, expiry can be varied.
- **Comparison** to analytical (Black-Scholes) and Python implementations.

## How to Run Benchmarks
1. Run the benchmark script:
   ```bash
   python scripts/benchmark_mc.py
   ```
   This will generate MLIR files for different path counts, run the full pipeline, and record execution times in `docs/benchmark_mc.csv`.

2. (Optional) Compare to Python/analytical implementations for reference.

3. Plot the results (e.g., with matplotlib or gnuplot):
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.read_csv('docs/benchmark_mc.csv')
   plt.plot(df['paths'], df['exec_time_sec'], marker='o')
   plt.xlabel('Paths')
   plt.ylabel('Execution Time (s)')
   plt.title('MC Pipeline Scaling')
   plt.xscale('log')
   plt.show()
   ```

## Notes
- Ensure your pipeline and runtime are compiled with optimizations for fair benchmarking.
- For reproducibility, fix random seeds if possible.
- Record system specs and compiler flags if publishing results.

---

This benchmarking workflow demonstrates the performance scaling of your custom MLIR dialect and pipeline, a key requirement for quant and HFT engineering roles.
