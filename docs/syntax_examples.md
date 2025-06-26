# DerivLab Syntax Examples

Below are example usages of the DerivLab dialect in MLIR for simulating asset prices and computing option payoffs.

## Simulate GBM Path
```mlir
%prices = derivlab.simulate_gbm steps = 100 : i32, dt = 0.01 : f64, mu = 0.05 : f64, sigma = 0.2 : f64 : tensor<100xf64>
```

## Compute Call Option Payoff
```mlir
%payoff = derivlab.payoff type = "call" : string, strike = 100.0 : f64, %prices : tensor<100xf64> : f64
```

## Compute Put Option Payoff
```mlir
%payoff = derivlab.payoff type = "put" : string, strike = 100.0 : f64, %prices : tensor<100xf64> : f64
```
