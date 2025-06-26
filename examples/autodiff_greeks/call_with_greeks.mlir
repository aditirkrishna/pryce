// DerivLab example: European Call Option with Autodiff for Greeks
// Simulates GBM, computes call payoff, averages, discounts, autodiff wrt spot

func.func @price_call(%spot: f64) -> f64 {
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
// After autodiff pass (e.g. enzyme-opt --autodiff @price_call --wrt-arg=0 --ret-derivative):
// func.func @dprice_dspot(%spot: f64) -> f64 { ... }
