// DerivLab example: Digital (binary) call option pricing
// Simulates GBM, computes digital call payoff, averages, discounts, returns price

func.func @price_digital_call() -> f64 {
  %spot = arith.constant 100.0 : f64
  %vol = arith.constant 0.25 : f64
  %rate = arith.constant 0.03 : f64
  %expiry = arith.constant 1.0 : f64
  %strike = arith.constant 100.0 : f64
  %npaths = arith.constant 50000 : i32
  %payout = arith.constant 1.0 : f64

  %paths = derivlab.simulate_gbm %spot, %vol, %rate, %expiry, %npaths : (f64, f64, f64, f64, i32) -> tensor<50000xf64>
  %payoffs = derivlab.payoff_digital_call %paths, %strike, %payout : (tensor<50000xf64>, f64, f64) -> tensor<50000xf64>
  %avg = derivlab.average %payoffs : tensor<50000xf64> -> f64
  %price = derivlab.discount %avg, %rate, %expiry : (f64, f64, f64) -> f64

  return %price : f64
}
