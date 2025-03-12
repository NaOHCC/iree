test:
	iree-compile --iree-hal-target-backends=cuda \
	--iree-codegen-reorder-workgroups-strategy=swizzle \
	--iree-codegen-reorder-workgroups-log-swizzle-tile=3 \
	--iree-llvmgpu-enable-prefetch \
	--iree-cuda-target=sm_80 \
	--mlir-print-ir-after-all \
	matmul.mlir \
	--mlir-disable-threading \
	--mlir-elide-elementsattrs-if-larger=10 \
	-o test.vmfb \
	2>log.mlir

tosa:
	iree-compile --iree-hal-target-backends=cuda \
	--iree-llvmgpu-enable-prefetch \
	--iree-cuda-target=sm_80 \
	--mlir-print-ir-after-all \
	tosa.mlir \
	--mlir-disable-threading \
	--mlir-elide-elementsattrs-if-larger=10 \
	-o test.vmfb \
	2>log.mlir

flow:
	iree-compile --iree-hal-target-backends=cuda \
	--iree-llvmgpu-enable-prefetch \
	--iree-cuda-target=sm_80 \
	--mlir-print-ir-after-all \
	flow.mlir \
	--mlir-disable-threading \
	--mlir-elide-elementsattrs-if-larger=10 \
	-o test.vmfb \
	2>log.mlir
