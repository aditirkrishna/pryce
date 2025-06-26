// Lowered MLIR (Standard/LLVM dialects) for Digital Option example
// Reference for FileCheck and debugging.
// CHECK: llvm.func @price_digital_call() -> f64

llvm.func @price_digital_call() -> f64 {
  // Allocate tensor for paths
  %0 = llvm.alloca ... : !llvm.ptr<tensor<50000xf64>>
  // S0 = 100.0
  %1 = llvm.constant(100.0 : f64) : f64
  // Loop: for i = 0 to 50000
  llvm.br ^bb1(%1 : f64)
^bb1(%S: f64):
  // ...simulate GBM step
  // ...store in %0
  // ...
  llvm.cond_br ...
  // Compute digital payoff: payout if S > strike else 0
  // ...
  // Average over all paths
  // ...
  // Discount: price = avg * exp(-rate * expiry)
  // ...
  llvm.return %final_price : f64
}
