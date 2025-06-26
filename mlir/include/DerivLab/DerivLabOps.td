//===- DerivLabOps.td - DerivLab Dialect Operations --------*- tablegen -*-===//
//
//  This file defines the DerivLab dialect operations for MLIR.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def DerivLab_Dialect : Dialect {
  let name = "derivlab";
  let cppNamespace = "DerivLab";
}

//===----------------------------------------------------------------------===//
// simulate_gbm: Simulates Geometric Brownian Motion paths for asset prices
//   steps: number of time steps (integer)
//   dt: time step size (float)
//   mu: drift (float)
//   sigma: volatility (float)
def SimulateGBMOp : DerivLab_Op<"simulate_gbm", [NoSideEffect]> {
  let summary = "Simulate a geometric Brownian motion path";
  let arguments = (ins
    I32Attr:$steps,   // Number of time steps
    F64Attr:$dt,      // Time step size
    F64Attr:$mu,      // Drift
    F64Attr:$sigma    // Volatility
  );
  let results = (outs F64Tensor);
}

//===----------------------------------------------------------------------===//
// payoff: Computes the payoff of an option (call or put)
//   type: option type (string: "call" or "put")
//   strike: strike price (float)
//   prices: asset prices (tensor)
def PayoffOp : DerivLab_Op<"payoff", [NoSideEffect]> {
  let summary = "Compute option payoff";
  let arguments = (ins
    StrAttr:$type,      // Option type
    F64Attr:$strike,    // Strike price
    F64Tensor:$prices   // Asset prices
  );
  let results = (outs F64Attr:$payoff);
}
