// MLIR test: call option example

module {
  // Simulate 100-step GBM path with mu=0.05, sigma=0.2
  %prices = derivlab.simulate_gbm steps = 100 : i32, dt = 0.01 : f64, mu = 0.05 : f64, sigma = 0.2 : f64 : tensor<100xf64>
  // Compute call option payoff with strike=100.0
  %payoff = derivlab.payoff type = "call" : string, strike = 100.0 : f64, %prices : tensor<100xf64> : f64
  // Expected: payoff = max(prices[-1] - strike, 0)
}
