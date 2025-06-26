// Integration test: simulate_gbm + payoff + (discount)
// CHECK-LABEL: module
// CHECK: derivlab.simulate_gbm
// CHECK: derivlab.payoff

module {
  %paths = derivlab.simulate_gbm steps = 5 : i32, dt = 0.02 : f64, mu = 0.03 : f64, sigma = 0.15 : f64 : tensor<5xf64>
  %call = derivlab.payoff type = "call" : string, strike = 100.0 : f64, %paths : tensor<5xf64> : f64
  // Optionally, discount result (if op exists)
  // %pv = derivlab.discount rate = 0.04 : f64, time = 1.0 : f64, value = %call : f64 : f64
}
