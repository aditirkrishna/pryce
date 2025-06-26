#!/usr/bin/env python3
"""
Fuzz testing script for DerivLab MLIR ops.
Generates random/edge-case simulate_gbm and payoff ops, writes to .mlir, and optionally runs verifier.
"""
import random
import subprocess
import os

TEST_DIR = os.path.join(os.path.dirname(__file__), '../test/fuzz')
MLIR_OPT = 'mlir-opt'

os.makedirs(TEST_DIR, exist_ok=True)

FUZZ_CASES = 20
for i in range(FUZZ_CASES):
    steps = random.choice([0, 1, 10, -5, 1000])
    dt = random.choice([0.0, 0.01, -0.01, 1.0])
    mu = random.uniform(-2, 2)
    sigma = random.choice([0.0, 0.2, -1.0, 100.0])
    strike = random.choice([0.0, 100.0, -50.0, 1e6])
    typ = random.choice(['call', 'put', 'invalid'])

    mlir = f"""module {{
  %paths = derivlab.simulate_gbm steps = {steps} : i32, dt = {dt} : f64, mu = {mu:.2f} : f64, sigma = {sigma} : f64 : tensor<5xf64>
  %payoff = derivlab.payoff type = \"{typ}\" : string, strike = {strike} : f64, %paths : tensor<5xf64> : f64
}}\n"""
    fname = os.path.join(TEST_DIR, f'fuzz_case_{i}.mlir')
    with open(fname, 'w') as f:
        f.write(mlir)
    # Optionally run mlir-opt to check for crashes or errors
    try:
        subprocess.run([MLIR_OPT, fname], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"[Fuzz {i}] Error or crash detected:\n", e.stderr.decode())
    else:
        print(f"[Fuzz {i}] Passed verifier.")
