from iree.compiler.ir import *
from iree.compiler.dialects import builtin, func, linalg, tensor

from iree.compiler.dialects.linalg.opdsl.lang import *
from iree.compiler.ir import *
import os

# from iree.compiler
import iree.compiler
from iree.compiler.dialects import (
    func,
    gpu,
    iree_codegen,
    iree_transform,
    linalg,
    pdl,
    transform,
)
from iree.compiler.dialects.bufferization import LayoutMapOption
from iree.compiler.dialects.transform import (
    bufferization,
    gpu as gpu_transform,
    interpreter as interp,
    loop,
    structured,
)


T1 = TV.T1
T2 = TV.T2

# interp.enable_debug()

ctx = Context()


@linalg_structured_op
def matmul_mono(
    A=TensorDef(T, S.M, S.K),
    B=TensorDef(T, S.K, S.N),
    C=TensorDef(T, S.M, S.N, output=True),
):
    domain(D.m, D.n, D.k)
    C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


FILE = "tmp/payload.mlir"
os.makedirs(os.path.dirname(FILE), exist_ok=True)


def run(payload: ir.Module):

    def _interpreter(f: Callable):
        m = f()
        m.operation.verify()
        print(f"Before transform, verify {payload.operation.verify()}:\n", payload)
        print("transform sequence:\n", m)
        interp.apply_named_sequence(payload, m.body.operations[0], m)
        print(f"After transform, verify {payload.operation.verify()}:\n", payload)
        with open(FILE, "w") as file:
            file.write(str(payload))

        return payload

    return _interpreter


def create_sequence(func: Callable) -> Callable:
    transform_root_module = r"""
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.yield
  }
}"""

    @functools.wraps(func)
    def decorated() -> ir.Module:
        with ctx, Location.unknown():
            transform_module = ir.Module.parse(transform_root_module)
            named_sequence_op = transform_module.body.operations[0]
            target = named_sequence_op.bodyTarget
            with InsertionPoint.at_block_begin(named_sequence_op.body):
                func(target)

            # transform_module.operation.verify()
            # interp.apply_named_sequence(
            #     payload_module, transform_module.body.operations[0], transform_module
            # )
        return transform_module

    return decorated


def build_payload():
    with ctx, Location.unknown():
        payload_module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(payload_module.body):

            @func.FuncOp.from_py_func(
                # RankedTensorType.get((4096, 4096), f32),
                # RankedTensorType.get((4096, 4096), f32),
                # RankedTensorType.get((4096, 4096), f32),
            )
            def test_matmul_mono():
                memref = MemRefType.get((4096, 4096), f32)
                token_ty = gpu.AsyncTokenType.get()
                t1 = gpu.wait(token_ty, [])
                a_device, t2 = gpu.alloc(memref, token_ty, [t1], [], [])
                b_device, t3 = gpu.alloc(memref, token_ty, [t2], [], [])
                c_device, t4 = gpu.alloc(memref, token_ty, [t3], [], [])
                t7 = gpu.wait(token_ty, [t4])
                matmul_mono(a_device, b_device, outs=[c_device])
                return

        # with open("template.mlir", "r") as file:
        #     payload_module = ir.Module.parse(file.read())

    return payload_module


def clean(target, op_name="func.func"):
    funcOp = structured.MatchOp.match_op_names(target, op_name)
    with InsertionPoint(transform.ApplyPatternsOp(funcOp).patterns):
        transform.ApplyCanonicalizationPatternsOp()
        structured.ApplyTilingCanonicalizationPatternsOp()

    transform.ApplyCommonSubexpressionEliminationOp(funcOp)


def getFuncOp(target):
    return structured.MatchOp.match_op_names(target, "func.func")


def getGpuFuncOp(target):
    return structured.MatchOp.match_op_names(target, "gpu.func")


# getFuncOp = getGpuFuncOp


def oneShotBufferize(target):
    return bufferization.OneShotBufferizeOp(
        target,
        function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap,
        bufferize_function_boundaries=True,
    )


def print_op(target, hint):
    transform.PrintOp(target=target, name=hint)


def get_block_dims(block_tile, reg_tile):
    block_dims: list[int] = []
    for block_tile_size, reg_tile_size in zip(block_tile, reg_tile):
        if (block_tile_size != 0) and (reg_tile_size != 0):
            block_dims.append(block_tile_size // reg_tile_size)
        else:
            Warning("block_tile_size or reg_tile_size is 0")
            block_dims.append(1)
    return block_dims


def march_op(source, target_op_name):
    return structured.MatchOp.match_op_names(source, target_op_name)


def get_index_attr(val: int) -> ir.IntegerAttr:
    return ir.IntegerAttr.get(ir.IndexType.get(), val)


def get_i64_attr(val: int) -> ir.IntegerAttr:
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), val)


@run(build_payload())
@create_sequence
def build(target):
    smem_space_str = "#gpu.memory_space<workgroup>"  # HACK: what different between memory_space and address_space?
    reg_space_str = "#gpu.memory_space<private>"
    # smem_space_str = "#gpu.address_space<workgroup>"
    # reg_space_str = "#gpu.address_space<private>"
    smem_space = ir.ArrayAttr.get([Attribute.parse(smem_space_str)])
    reg_space = ir.ArrayAttr.get([Attribute.parse(reg_space_str)])
    tiling_level_2 = [128, 64, 0]
    tiling_level_1 = [16, 4, 0]
    block_dims = get_block_dims(tiling_level_2, tiling_level_1)
    CopyToWorkgroupMemoryMarker = {
        "key": "__internal_linalg_transform__",
        "value": StringAttr.get("copy_to_workgroup_memory"),
    }
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.None_
    )
    TranslationInfo = {
        "key": "translation_info",
        "value": iree_codegen.TranslationInfoAttr.get(
            pipeline_attr,
            None,
            block_dims,
            None,
            DictAttr.get(
                {
                    "pipeline": get_i64_attr(3),
                    "store_stage": get_i64_attr(1),
                }
            ),
        ),
    }

    reduce_tile = [0, 0, 8]
    # bufferize
    # target = getFuncOp(target)
    bufferizeOp = oneShotBufferize(target)
    target = bufferizeOp.transformed

    generic = structured.MatchOp.match_op_names(target, "linalg.generic")
    block_mapping = Attribute.parse("[ #gpu.block<x>, #gpu.block<y> ]")
    thread_mapping = Attribute.parse("[ #gpu.thread<x>, #gpu.thread<y> ]")

    # return
    level_2_op = structured.TileUsingForallOp(
        # transform.OperationType.get("linalg.generic"),  # tiled_op_type
        # transform.OperationType.get("scf.forall"),  # loops_type
        generic,
        # num_threads=[2, 4, 4],
        tile_sizes=tiling_level_2,
        mapping=block_mapping,
    )

    transform.PrintOp(target=target, name="before promote C")

    promoted = structured.PromoteOp(
        transformed=transform.AnyOpType.get(),
        target=level_2_op.tiled_op,
        operands_to_promote=[2],
        mapping=smem_space,
    )
    print_op(promoted.transformed, "after promote C")
    redution_tiled = structured.TileUsingForOp(
        promoted.transformed,
        sizes=reduce_tile,
        # num_threads=reduce_tile
    )
    # clean(target)
    transform.PrintOp(target=target, name="before promote A and B")
    input_promoted = structured.PromoteOp(
        transformed=transform.AnyOpType.get(),
        target=redution_tiled.tiled_linalg_op,
        operands_to_promote=[0, 1],
        mapping=smem_space,
    )
    clean(target)
    print_op(target, "after promote A and B")

    level_1_op = structured.TileUsingForallOp(
        transform.OperationType.get("linalg.generic"),  # tiled_op_type
        transform.OperationType.get("scf.forall"),  # loops_type
        input_promoted.transformed,
        # num_threads=[2, 4, 4],
        tile_sizes=tiling_level_1,
        mapping=thread_mapping,
    )

    # mark copy to workgroup memory
    memref_copy = march_op(target, "memref.copy")
    iree_transform.AddAttrbuiteOp([memref_copy], **CopyToWorkgroupMemoryMarker)
    transform.ApplyRegisteredPassOp(
        transform.AnyOpType.get(),
        getFuncOp(target),
        "iree-codegen-memrefcopy-to-linalg",
    )

    # parent_op = transform.GetParentOp(
    #     transform.AnyOpType.get(), level_1_op.tiled_op, isolated_from_above=True
    # )
    print_op(target, "before map nested forall to threads")
    gpu_launch_op = gpu_transform.MapForallToBlocks(
        getFuncOp(target), generate_gpu_launch=True
    )

    gpu_launch_op = gpu_transform.MapNestedForallToThreads(
        gpu_launch_op.result, block_dims=block_dims
    )
    print_op(target, "after map nested forall to threads")

    # gpu-kernel-outlining
    outlined = transform.ApplyRegisteredPassOp(
        transform.AnyOpType.get(), target, "gpu-kernel-outlining"
    )
    gpu_module = march_op(outlined.result, "gpu.module")
    clean(outlined.result, "gpu.func")
    # DistributeSharedMemoryCopy

    iree_transform.AddAttrbuiteOp([getGpuFuncOp(gpu_module)], **TranslationInfo)
    print_op(getGpuFuncOp(gpu_module), "before distribute shared memory copy")
    iree_transform.GpuDistributeSharedMemoryCopyOp(getGpuFuncOp(gpu_module))
    clean(gpu_module, "gpu.func")
    print_op(getGpuFuncOp(gpu_module), "after distribute shared memory copy")

    # iree_transform.ReduceSharedMemoryBankConflictsOp(getFuncOp(target))
    # clean(outlined.result, "gpu.func")
    # print_op(getGpuFuncOp(gpu_module), "before vectorize children")
    # structured.VectorizeChildrenAndApplyPatternsOp(getGpuFuncOp(gpu_module))
    # # clean(target)
    # print_op(getGpuFuncOp(gpu_module), "after vectorize children")

    # transform.ApplyRegisteredPassOp(
    #     transform.AnyOpType.get(),
    #     getGpuFuncOp(gpu_module),
    #     "iree-codegen-gpu-pipelining",
    # )
    transform.ApplyRegisteredPassOp(
        transform.AnyOpType.get(),
        getGpuFuncOp(gpu_module),
        "iree-codegen-gpu-reduce-bank-conflicts",
        options="padding-bits=32",
    )
    transform.ApplyRegisteredPassOp(
        transform.AnyOpType.get(),
        getGpuFuncOp(gpu_module),
        "iree-llvmgpu-vector-lowering",
    )

    # forop = march_op(target, "scf.for")
    # forop = transform.CastOp(transform.OperationType.get("scf.for"), forop)
    # loop.LoopPipelineOp(
    #     transform.AnyOpType.get(),
    #     forop,
    #     iteration_interval=1,
    # )

    # print_op(linalg_copy, "linalg_copy")
    # clean(target)
    # # generic = structured.MatchOp.match_op_names(target, "linalg.generic")
    # # transform.PrintOp(target=generic, name="xxx")
    # # clean(target)

    return
