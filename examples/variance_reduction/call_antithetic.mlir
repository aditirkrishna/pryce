func.func @price_call_with_antithetic() -> f64 {
  %spot = const.f64 100.0
  %vol = const.f64 0.2
  %rate = const.f64 0.05
  %expiry = const.f64 1.0
  %strike = const.f64 100.0
  %npaths = const.i32 50000

  %paths = derivlab.simulate_gbm_antithetic %spot, %vol, %rate, %expiry, %npaths : ...
  %payoffs = derivlab.payoff_call %paths, %strike : ...
  %avg = derivlab.average %payoffs : ...
  %price = derivlab.discount %avg, %rate, %expiry : ...
  return %price : f64
}
