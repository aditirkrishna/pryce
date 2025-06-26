//===- Passes.cpp - Register DerivLab lowering passes ---------------------===//
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "DerivLab/DerivLabDialect.h"

using namespace mlir;
using namespace DerivLab;

void registerDerivLabPasses() {
  // TODO: Register LowerSimulateGBM, LowerPayoff, LowerDiscount passes
  // Example:
  // PassRegistration<LowerSimulateGBMPass>("lower-derivlab-simulate-gbm", "Lower simulate_gbm ops");
  // PassRegistration<LowerPayoffPass>("lower-derivlab-payoff", "Lower payoff ops");
  // PassRegistration<LowerDiscountPass>("lower-derivlab-discount", "Lower discount ops");
}
