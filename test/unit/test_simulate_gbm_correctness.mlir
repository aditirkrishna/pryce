// Test simulation correctness: known GBM path
// S0 = 100, mu = 0.05, sigma = 0.2, dt = 1, steps = 1, Z = 0.0 (deterministic)
// Expected: S1 = S0 * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
//          = 100 * exp((0.05 - 0.5*0.04)*1 + 0.2*1*0.0)
//          = 100 * exp(0.03) ≈ 103.045
// CHECK: arith.constant 103.045

module {
  // Use a deterministic Z=0.0 for test
  %path = derivlab.simulate_gbm steps = 1 : i32, dt = 1.0 : f64, mu = 0.05 : f64, sigma = 0.2 : f64 : tensor<1xf64>
  // The lowering and runtime should produce a value ≈ 103.045
}
