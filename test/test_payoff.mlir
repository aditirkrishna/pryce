// MLIR test: payoff

// Unit test for payoff operation
// CHECK-LABEL: module
// CHECK: derivlab.payoff

module {
  // Provide constant prices for testing
  %prices = arith.constant dense<[110.0, 95.0, 120.0]> : tensor<3xf64>

  // Compute call payoff with strike=100.0
  %call = derivlab.payoff type = "call" : string, strike = 100.0 : f64, %prices : tensor<3xf64> : f64
  // CHECK: type = "call"
  // CHECK: strike = 100.0

  // Compute put payoff with strike=100.0
  %put = derivlab.payoff type = "put" : string, strike = 100.0 : f64, %prices : tensor<3xf64> : f64
  // CHECK: type = "put"
  // CHECK: strike = 100.0

  // Invalid: missing prices (should trigger verifier error)
  // expected-error@+1 {{payoff op requires prices input}}
  %bad = derivlab.payoff type = "call" : string, strike = 100.0 : f64 : f64
}
