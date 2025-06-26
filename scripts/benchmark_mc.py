#!/usr/bin/env python3
"""
Benchmark the MC pricing pipeline for different path counts and parameters.
Measures execution time and outputs CSV for plotting.
"""
import subprocess
import time
import csv

# Example path counts and parameters
path_counts = [100, 1000, 10000, 100000]
mu = 0.05
sigma = 0.2
r = 0.05
strike = 100.0
T = 1.0

with open('../docs/benchmark_mc.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['paths', 'exec_time_sec'])
    for n in path_counts:
        # Generate MLIR file for this n (pseudo-code, assumes template)
        mlir_file = f'test_bench_{n}.mlir'
        with open(mlir_file, 'w') as f:
            f.write(f"""
module {{
  %paths = derivlab.simulate_gbm steps = {n} : i32, dt = 1.0 : f64, mu = {mu} : f64, sigma = {sigma} : f64 : tensor<{n}xf64>
  %call = derivlab.payoff type = \"call\" : string, strike = {strike} : f64, %paths : tensor<{n}xf64> : f64
}}
""")
        # Run MLIR pipeline (assumes test_pipeline.sh accepts file arg)
        start = time.time()
        subprocess.run(['../scripts/test_pipeline.sh', mlir_file], check=True)
        end = time.time()
        writer.writerow([n, end - start])
print('Benchmark results written to docs/benchmark_mc.csv')
