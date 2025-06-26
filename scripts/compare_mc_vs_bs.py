#!/usr/bin/env python3
"""
Compare Monte Carlo (MLIR pipeline) vs Black-Scholes analytical pricing.
- Runs MC price(s) from CSV or text output.
- Computes analytical price using runtime/black_scholes.cpp (via ctypes or subprocess).
- Outputs CSV with MC, analytical, and error margin.
"""
import csv
import math
import ctypes
import os

# Load Black-Scholes C++ library (assume compiled as shared lib, e.g. libblack_scholes.so)
# For Windows, use .dll; for Linux, .so
LIBNAME = os.path.join(os.path.dirname(__file__), '../runtime/black_scholes.so')
bs_lib = ctypes.CDLL(LIBNAME)
bs_call = bs_lib.black_scholes_call
bs_call.restype = ctypes.c_double
bs_call.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]

# Example MC output CSV (columns: S, K, r, sigma, T, MC)
with open('../docs/mc_vs_bs_input.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    with open('../docs/mc_vs_bs_comparison.csv', 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(['S', 'K', 'r', 'sigma', 'T', 'MC', 'BS', 'abs_error'])
        for row in reader:
            S = float(row['S'])
            K = float(row['K'])
            r = float(row['r'])
            sigma = float(row['sigma'])
            T = float(row['T'])
            MC = float(row['MC'])
            BS = bs_call(S, K, r, sigma, T)
            abs_err = abs(MC - BS)
            writer.writerow([S, K, r, sigma, T, MC, BS, abs_err])
print('Comparison written to docs/mc_vs_bs_comparison.csv')
