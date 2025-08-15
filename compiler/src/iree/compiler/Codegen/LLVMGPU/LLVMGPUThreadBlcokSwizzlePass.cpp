// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUTHREADBLCOKSWIZZLEPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

template <typename OpTy>
static void createForAllDimensions(OpBuilder &builder, Location loc,
                                   SmallVectorImpl<Value> &values) {
  for (auto dim : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z})
    values.push_back(builder.create<OpTy>(loc, builder.getIndexType(), dim));
}

struct LLVMGPUThreadBlcokSwizzlePass final
    : impl::LLVMGPUThreadBlcokSwizzlePassBase<LLVMGPUThreadBlcokSwizzlePass> {

  using impl::LLVMGPUThreadBlcokSwizzlePassBase<
      LLVMGPUThreadBlcokSwizzlePass>::LLVMGPUThreadBlcokSwizzlePassBase;
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();
    IRRewriter rewriter(context);

    Value blockIdX;
    Value blockIdY;

    funcOp->walk([&](gpu::BlockIdOp idOp) {
      if (idOp.getDimension() == gpu::Dimension::x) {
        blockIdX = idOp.getResult();
      } else if (idOp.getDimension() == gpu::Dimension::y) {
        blockIdY = idOp.getResult();
      }
    });

    if (!blockIdX || !blockIdY) {
      return;
    }

    rewriter.setInsertionPointToStart(&funcOp.getBlocks().front());
    auto loc = funcOp.getLoc();
    SmallVector<Value> idOps;
    SmallVector<Value> dimOps;
    createForAllDimensions<gpu::BlockIdOp>(rewriter, funcOp->getLoc(), idOps);
    createForAllDimensions<gpu::GridDimOp>(rewriter, funcOp->getLoc(), dimOps);

    Value panelWidthValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(panelWidth));

    AffineExpr x, y, z, w;
    bindSymbols(context, x, y, z, w);
    auto blockIdx = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(0, 3, {x + y * z}, context),
        ValueRange{idOps[0], idOps[1], dimOps[0]});
    auto gridSize = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(0, 2, {x * y}, context),
        ValueRange{dimOps[0], dimOps[1]});

    auto panelSize = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(0, 1, {panelWidth * x}, context),
        ValueRange{dimOps[0]});
    auto panelOffset = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(0, 2, {x % y}, context),
        ValueRange{blockIdx, panelSize});
    auto panelIdx = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(0, 2, {x.floorDiv(y)}, context),
        ValueRange{blockIdx, panelSize});

    auto totalPanel = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(0, 2, {x.floorDiv(y)}, context),
        ValueRange{gridSize, panelSize});

    auto nextPanel = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(0, 1, {x + 1}, context), ValueRange{panelIdx});
    auto isLastPanel = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, nextPanel, totalPanel);
    auto ifOp = rewriter.create<scf::IfOp>(
        loc, isLastPanel,
        [&](OpBuilder builder, Location Loc) {
          Value v = rewriter.create<affine::AffineApplyOp>(
              loc, AffineMap::get(0, 4, {(x - y * z).floorDiv(w)}, context),
              ValueRange{gridSize, panelIdx, panelSize, dimOps[0]});
          builder.create<scf::YieldOp>(loc, v);
        },
        [&](OpBuilder builder, Location Loc) {
          builder.create<scf::YieldOp>(loc, panelWidthValue);
        });
    auto stride = ifOp->getResult(0);

    Value one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    Value isOdd = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        rewriter.create<arith::AndIOp>(loc, blockIdx, one), one);
    auto ifOp2 = rewriter.create<scf::IfOp>(
        loc, isOdd,
        [&](OpBuilder builder, Location Loc) {
          Value col = rewriter.create<affine::AffineApplyOp>(
              loc, AffineMap::get(0, 3, {x - 1 - (y.floorDiv(z))}, context),
              ValueRange{dimOps[0], panelOffset, stride});
          builder.create<scf::YieldOp>(loc, col);
        },
        [&](OpBuilder builder, Location Loc) {
          Value col = rewriter.create<affine::AffineApplyOp>(
              loc, AffineMap::get(0, 2, {x.floorDiv(y)}, context),
              ValueRange{panelOffset, stride});
          builder.create<scf::YieldOp>(loc, col);
        });

    auto col = ifOp2->getResult(0);
    auto row = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(0, 4, {x % y + z * w}, context),
        ValueRange{panelOffset, stride, panelIdx, panelWidthValue});

    // Replace the blockIdX and blockIdY with the new values
    blockIdX.replaceAllUsesWith(col);
    blockIdY.replaceAllUsesWith(row);
  }
};

} // namespace
} // namespace mlir::iree_compiler
