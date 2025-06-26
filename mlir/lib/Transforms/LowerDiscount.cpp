//===- LowerDiscount.cpp - Lower DerivLab discount op to math ops ---------===//
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "DerivLab/DerivLabDialect.h"
#include <cmath>

using namespace mlir;
using namespace DerivLab;

namespace {
struct LowerDiscountPattern : public OpRewritePattern<DiscountOp> {
  using OpRewritePattern<DiscountOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DiscountOp op, PatternRewriter &rewriter) const override {
    auto rate = op.getRate();
    auto time = op.getTime();
    auto value = op.getValue();
    // Lower discount to: pv = value * exp(-rate * time)
    auto negRateTime = rewriter.create<arith::MulFOp>(op.getLoc(),
      rewriter.create<arith::NegFOp>(op.getLoc(), rate), time);
    auto exp = rewriter.create<math::ExpOp>(op.getLoc(), negRateTime);
    auto pv = rewriter.create<arith::MulFOp>(op.getLoc(), value, exp);
    rewriter.replaceOp(op, pv.getResult());
    return success();
  }
};
} // namespace

void populateLowerDiscountPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerDiscountPattern>(patterns.getContext());
}
