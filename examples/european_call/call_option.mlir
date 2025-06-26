// DerivLab example: European Call Option Pricing
// Simulates GBM, computes call payoff, averages, discounts, returns price

func.func @price_call() -> f64 {
  %spot = arith.constant 100.0 : f64
  %vol = arith.constant 0.2 : f64
  %rate = arith.constant 0.05 : f64
  %expiry = arith.constant 1.0 : f64
  %strike = arith.constant 105.0 : f64
  %npaths = arith.constant 100000 : i32

  %paths = derivlab.simulate_gbm %spot, %vol, %rate, %expiry, %npaths : (f64, f64, f64, f64, i32) -> tensor<100000xf64>
  %payoffs = derivlab.payoff_call %paths, %strike : (tensor<100000xf64>, f64) -> tensor<100000xf64>
  %avg = derivlab.average %payoffs : tensor<100000xf64> -> f64
  %price = derivlab.discount %avg, %rate, %expiry : (f64, f64, f64) -> f64

  return %price : f64
}
