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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUCONVERTSTATICSHAREDMEMORYALLOCPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct LLVMGPUConvertStaticSharedMemoryAllocPass final
    : impl::LLVMGPUConvertStaticSharedMemoryAllocPassBase<
          LLVMGPUConvertStaticSharedMemoryAllocPass> {

  using impl::LLVMGPUConvertStaticSharedMemoryAllocPassBase<
      LLVMGPUConvertStaticSharedMemoryAllocPass>::
      LLVMGPUConvertStaticSharedMemoryAllocPassBase;
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    // first, collect all the static shared memory alloc ops
    SmallVector<memref::AllocOp> allocOps;
    IRMapping mapping;
    funcOp->walk([&](memref::AllocOp allocOp) {
      auto type = allocOp.getType();
      if (!type.hasStaticShape() && !hasSharedMemoryAddressSpace(type))
        return;

      // auto elNumBytes = allocOp.getType().getElementTypeBitWidth() / 8;
      // auto numBytes = type.getNumElements() * elNumBytes;
      // totalAllocBytes += numBytes;

      allocOps.push_back(allocOp);
    });

    if (allocOps.empty())
      return;

    // second, create dynamic shared memory op
    rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
    auto type = MemRefType::get(
        ShapedType::kDynamic, rewriter.getI8Type(), MemRefLayoutAttrInterface{},
        gpu::AddressSpaceAttr::get(
            rewriter.getContext(),
            gpu::GPUDialect::getWorkgroupAddressSpace()));
    auto dynamicSharedMemoryOp =
        (rewriter.create<gpu::DynamicSharedMemoryOp>(funcOp.getLoc(), type));

    // third, replace the static shared memory alloc ops with views into the
    // dynamic shared memory op, remove address space
    long currentOffset = 0;
    // TODO: use bestfit allocation strategy
    for (auto &allocOp : llvm::make_early_inc_range(allocOps)) {
      auto allocType = llvm::cast<MemRefType>(allocOp.getType());
      rewriter.setInsertionPoint(allocOp);

      Value sourceOp = dynamicSharedMemoryOp;
      // if (enableDropAddressSpace) {
      //   allocType =
      //       MemRefType::get(allocType.getShape(), allocType.getElementType(),
      //                       MemRefLayoutAttrInterface{});
      //   auto normalI8Type =
      //       MemRefType::get(ShapedType::kDynamic, rewriter.getI8Type(),
      //                       MemRefLayoutAttrInterface{});

      //   sourceOp = rewriter.create<memref::MemorySpaceCastOp>(
      //       dynamicSharedMemoryOp.getLoc(), normalI8Type,
      //       dynamicSharedMemoryOp);
      // }

      auto elNumBytes = allocOp.getType().getElementTypeBitWidth() / 8;
      auto numBytes = allocType.getNumElements() * elNumBytes;
      auto offsetValue = rewriter.create<arith::ConstantIndexOp>(
          allocOp.getLoc(), currentOffset);
      currentOffset += numBytes;
      auto viewOp = rewriter.create<memref::ViewOp>(
          allocOp->getLoc(), allocType, sourceOp, offsetValue,
          /* sizes */ ArrayRef<Value>({}));
      auto dims = allocType.getShape();
      auto lastDim = *dims.end();
      if ((lastDim * elNumBytes) % 4 == 0)
        rewriter.create<memref::AssumeAlignmentOp>(allocOp->getLoc(), viewOp,
                                                   16);
      else if ((lastDim * elNumBytes) % 2 == 0)
        rewriter.create<memref::AssumeAlignmentOp>(allocOp->getLoc(), viewOp,
                                                   8);

      rewriter.replaceOp(allocOp, viewOp);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
