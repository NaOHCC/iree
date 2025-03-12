#!/bin/bash

# 定义替换的列表
replacements=(
# "LLVMGPUDefault"
# "LLVMGPUBaseLowering"
"LLVMGPUDistribute"
# "LLVMGPUVectorize"
"LLVMGPUMatmulSimt"
"LLVMGPUMatmulTensorCore"
"LLVMGPUTransposeSharedMem"
"LLVMGPUWarpReduction"
# "LLVMGPUPackUnPack"
"LLVMGPUMatmulTensorCoreMmaSync"
# "LLVMGPUVectorDistribute"
# "LLVMGPUPadAndVectorDistribute"
# "LLVMGPUWinogradVectorize"
# "LLVMGPUTileAndFuse"
)


# 依次替换 {placeholder} 并追加到 text
for replacement in "${replacements[@]}"; do
# 读取文件内容
template=$(<matmul.mlir)

# 初始化结果变量
text=""
  replaced_text=${template//\{placeholder\}/$replacement}
  text+="$replaced_text\n"

# 去掉最后的换行符
text=$(echo -e "$text" | sed '/^$/d')

# 打印结果（可选）
# echo "$text"

# 将 text 写入一个临时文件
temp_file=$(mktemp)
echo -e "$text" > "$temp_file"

echo "Compiling: $replacement"
iree-compile --iree-hal-target-backends=cuda --iree-cuda-target=sm_80 --mlir-print-ir-after-all \
--iree-llvmgpu-enable-prefetch \
--mlir-disable-threading --mlir-elide-elementsattrs-if-larger=10 \
-o test.vmfb 2>$replacement.mlir "$temp_file"

# 删除临时文件
rm "$temp_file"
echo "Done: $replacement"
done