//===- DerivLabDialect.cpp - DerivLab Dialect Implementation ----*- C++ -*-===//
//
//  This file implements the DerivLab MLIR dialect.
//===----------------------------------------------------------------------===//

#include "DerivLab/DerivLabDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cassert>
#include <iostream>

using namespace mlir;
using namespace DerivLab;

DerivLabDialect::DerivLabDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<DerivLabDialect>()) {
  initialize();
  // Example assertion: ensure context is valid
  assert(context && "MLIRContext must not be null");

  // Example logging: dialect loaded
  llvm::errs() << "[DerivLabDialect] Loaded DerivLab dialect\n";
}

// Register all ops (simulate_gbm, payoff, discount)
void DerivLabDialect::initialize() {
  addOperations<
    SimulateGBMOp,
    PayoffOp,
    DiscountOp
  >();
}
