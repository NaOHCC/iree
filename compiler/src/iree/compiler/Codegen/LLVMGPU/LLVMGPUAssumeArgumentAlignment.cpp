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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUASSUMEARGUMENTALIGNMENTPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct LLVMGPUAssumeArgumentAlignmentPass final
    : impl::LLVMGPUAssumeArgumentAlignmentPassBase<
          LLVMGPUAssumeArgumentAlignmentPass> {

  using impl::LLVMGPUAssumeArgumentAlignmentPassBase<
      LLVMGPUAssumeArgumentAlignmentPass>::
      LLVMGPUAssumeArgumentAlignmentPassBase;
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
    auto arguments = funcOp.getArguments();
    for (auto arg : arguments) {
      auto type = llvm::dyn_cast_if_present<MemRefType>(arg.getType());
      if (!type && !type.hasStaticShape())
        continue;
      rewriter.create<memref::AssumeAlignmentOp>(funcOp->getLoc(), arg,
                                                 alignment);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
