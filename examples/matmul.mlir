#compilation0 = #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 64, 8]]>,
  translation_info = <pipeline = {placeholder} workgroup_size = [16, 8, 1]
  ,{ pipeline_depth = 3,   store_stage = 1 }>
  >
  // translation_info = <LLVMGPUDefault workgroup_size = [64, 2, 1]
  // translation_info = <LLVMGPUBaseLowering workgroup_size = [64, 2, 1]
  // translation_info = <LLVMGPUDistribute workgroup_size = [64, 2, 1]
  // translation_info = <LLVMGPUVectorize workgroup_size = [64, 2, 1]
  // translation_info = <LLVMGPUMatmulSimt workgroup_size = [64, 2, 1]
  // translation_info = <LLVMGPUMatmulTensorCore workgroup_size = [64, 2, 1]
  // translation_info = <LLVMGPUTransposeSharedMem workgroup_size = [64, 2, 1]
  // translation_info = <LLVMGPUWarpReduction workgroup_size = [64, 2, 1]
  // translation_info = <LLVMGPUPackUnPack workgroup_size = [64, 2, 1]
  // translation_info = <LLVMGPUMatmulTensorCoreMmaSync workgroup_size = [64, 2, 1]
  // translation_info = <LLVMGPUVectorDistribute workgroup_size = [64, 2, 1]
  // translation_info = <LLVMGPUPadAndVectorDistribute workgroup_size = [64, 2, 1]
  // translation_info = <LLVMGPUWinogradVectorize workgroup_size = [64, 2, 1]
  // translation_info = <LLVMGPUTileAndFuse workgroup_size = [64, 2, 1]

func.func @matmul(%lhs: tensor<4096x4096xf32>, %rhs: tensor<4096x4096xf32>, %acc: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
  %result = linalg.matmul {compilation_info = #compilation0} ins(%lhs, %rhs: tensor<4096x4096xf32>, tensor<4096x4096xf32>) outs(%acc: tensor<4096x4096xf32>) -> tensor<4096x4096xf32>

  return %result: tensor<4096x4096xf32>
}