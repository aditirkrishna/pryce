//===- DiscountOp.td - Discount Operation for DerivLab --------*- tablegen -*-===//
// Defines the discount operation for present value calculation.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def DiscountOp : DerivLab_Op<"discount", [NoSideEffect]> {
  let summary = "Discounts a value to present value";
  let arguments = (ins
    F64Attr:$rate,    // Discount rate
    F64Attr:$time,    // Time to expiry
    F64Attr:$value    // Value to discount
  );
  let results = (outs F64Attr:$pv);
}
