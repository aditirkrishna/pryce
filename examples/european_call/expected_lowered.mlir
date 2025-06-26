// Lowered MLIR (Standard/LLVM dialects) for European Call Option example
// This is a reference for FileCheck and debugging.
// CHECK: llvm.func @price_call() -> f64

llvm.func @price_call() -> f64 {
  // Allocate tensor for paths
  %0 = llvm.alloca ... : !llvm.ptr<tensor<100000xf64>>
  // S0 = 100.0
  %1 = llvm.constant(100.0 : f64) : f64
  // Loop: for i = 0 to 100000
  llvm.br ^bb1(%1 : f64)
^bb1(%S: f64):
  // ...simulate GBM step: S = S * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
  // ...store in %0
  // ...
  llvm.cond_br ...
  // Compute payoff: max(S - strike, 0.0)
  // ...
  // Average over all paths
  // ...
  // Discount: price = avg * exp(-rate * expiry)
  // ...
  llvm.return %final_price : f64
}
