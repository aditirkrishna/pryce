// Lowered MLIR for autodiff Greeks example
// Shows primal and autodiff (gradient) function after lowering
// CHECK: llvm.func @price_call(%spot: f64) -> f64
// CHECK: llvm.func @dprice_dspot(%spot: f64) -> f64

llvm.func @price_call(%spot: f64) -> f64 {
  // ...original pricing logic lowered to LLVM dialect...
  llvm.return %price : f64
}
llvm.func @dprice_dspot(%spot: f64) -> f64 {
  // ...autodiff gradient logic lowered to LLVM dialect...
  llvm.return %delta : f64
}
