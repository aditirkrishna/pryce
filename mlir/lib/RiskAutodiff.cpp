// RiskAutodiff.cpp: Adjoint/autodiff lowering for risk hooks
#include "DerivLab/RiskDSL.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;

namespace {
struct LowerRiskAutodiffPattern : public OpRewritePattern<Risk_ComputeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Risk_ComputeOp op, PatternRewriter &rewriter) const override {
    // Lower risk.compute to autodiff/adjoint kernels for Greeks
    // ...
    return success();
  }
};
} // namespace

void populateRiskAutodiffPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerRiskAutodiffPattern>(patterns.getContext());
}
