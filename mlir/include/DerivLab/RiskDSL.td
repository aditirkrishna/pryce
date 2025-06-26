// TableGen: Symbolic Risk DSL Ops for DerivLab
// Defines contract and risk ops for symbolic-to-executable risk engine

def Contract_DigitalOptionOp : DerivLab_Op<"contract.call_digital"> {
  let summary = "Symbolic contract for digital option";
  let arguments = (ins StringAttr:$asset, F64Attr:$strike, F64Attr:$payout, F64Attr:$expiry);
  let results = (outs None);
}

def Risk_ComputeOp : DerivLab_Op<"risk.compute"> {
  let summary = "Risk computation hooks (greeks, scenarios)";
  let arguments = (ins ArrayAttr:$greeks, ArrayAttr:$scenarios);
  let results = (outs None);
}
