// Unit test for simulate_gbm operation
// CHECK-LABEL: module
// CHECK: derivlab.simulate_gbm

module {
  // Valid parameters
  %prices = derivlab.simulate_gbm steps = 10 : i32, dt = 0.01 : f64, mu = 0.05 : f64, sigma = 0.2 : f64 : tensor<10xf64>
  // CHECK: steps = 10
  // CHECK: dt = 0.01
  // CHECK: mu = 0.05
  // CHECK: sigma = 0.2

  // Invalid: negative sigma (should trigger verifier error)
  // expected-error@+1 {{sigma must be positive}}
  %bad = derivlab.simulate_gbm steps = 10 : i32, dt = 0.01 : f64, mu = 0.05 : f64, sigma = -1.0 : f64 : tensor<10xf64>
}
