// Unit test for discount operation (example, assuming discount op exists)
// CHECK-LABEL: module
// CHECK: derivlab.discount

module {
  // Valid: discount a value to present value
  %pv = derivlab.discount rate = 0.05 : f64, time = 1.0 : f64, value = 100.0 : f64 : f64
  // CHECK: rate = 0.05
  // CHECK: time = 1.0
  // CHECK: value = 100.0

  // Invalid: negative time (should trigger verifier error)
  // expected-error@+1 {{time must be non-negative}}
  %bad = derivlab.discount rate = 0.05 : f64, time = -1.0 : f64, value = 100.0 : f64 : f64
}
