# ===- run-module-gpu.py --------------------------------------------------===//
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ===----------------------------------------------------------------------===//
#
#  This file is a script to test whether the specified MLIR module on the GPU
#  calculates the same result as NumPy.
#
# ===----------------------------------------------------------------------===//
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import mlir.dialects.func as func
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir import runtime as rt
from mlir.ir import *
import argparse as ap
import ctypes

import cupy as cp

import numpy as np


def new_ranked_memref_descriptor(nparray: np.ndarray):
    if nparray.dtype == "bfloat16":
        ctp = rt.F16
    else:
        ctp = rt.as_ctype(nparray.dtype)

    if nparray.ndim == 0:
        x = rt.make_zero_d_memref_descriptor(ctp)()
        x.allocated = nparray.ctypes.data
        x.aligned = nparray.ctypes.data_as(ctypes.POINTER(ctp))
        x.offset = ctypes.c_longlong(0)
        return x

    x = rt.make_nd_memref_descriptor(nparray.ndim, ctp)()
    nbytes = nparray.nbytes
    buffer = ctypes.create_string_buffer(nbytes)
    ctypes.memmove(buffer, nparray.ctypes.data, nbytes)
    x.allocated = ctypes.cast(buffer, ctypes.c_void_p).value
    x.aligned = ctypes.cast(buffer, ctypes.POINTER(ctp))
    x.offset = ctypes.c_longlong(0)
    x.shape = nparray.ctypes.shape

    # Numpy uses byte quantities to express strides, MLIR OTOH uses the
    # torch abstraction which specifies strides in terms of elements.
    strides_ctype_t = ctypes.c_longlong * nparray.ndim
    x.strides = strides_ctype_t(*[x // nparray.itemsize for x in nparray.strides])
    return x


def get_memref_descriptors(args: list[Type]):
    memref_ptrs = []
    for arg in args:
        elem_type = to_numpy(str(arg.element_type))
        np_arg = np.random.rand(*arg.shape).astype(elem_type)
        memref_ptrs.append(
            ctypes.pointer(ctypes.pointer(new_ranked_memref_descriptor(np_arg)))
        )
    return memref_ptrs


M = 4096
N = M
K = M

# def make_ctypes_pointer(l):


def get_cuda_pointer():
    npy_dtype = np.float32
    A = np.random.randint(0, 10, (M, K)).astype(npy_dtype)
    B = np.random.randint(0, 10, (K, N)).astype(npy_dtype)
    C = np.zeros((M, N)).astype(npy_dtype)
    dA = cp.asarray(A)
    dB = cp.asarray(B)
    dC = cp.asarray(C)
    ptrs = [ctypes.pointer(p) for p in (dA.data.ptr, dB.data.ptr, dC.data.ptr)]
    return ptrs


def test(target, llvm_dir: str):
    with Context() as ctx:
        with open(target, "r") as file:
            newModule = Module.parse(file.read())

        engine = ExecutionEngine(
            newModule,
            shared_libs=[
                llvm_dir + "/lib/libmlir_c_runner_utils.so",
                llvm_dir + "/lib/libmlir_async_runtime.so",
                llvm_dir + "/lib/libmlir_runner_utils.so",
                llvm_dir + "/lib/libmlir_cuda_runtime.so",
            ],
            opt_level=3,
        )
        rf = engine.raw_lookup("test_matmul_mono")
        prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        packed_args = (ctypes.c_void_p * len([]))()
        f = prototype(rf)
        # ptrs = get_cuda_pointer()
        f(packed_args)
        print("finished")
        # print(cp.cuda.runtime.get_last_error())
        # engine.invoke(funcName, *memref_ptrs)
        # out = rt.ranked_memref_to_numpy(memref_ptrs[0][0])
        # if str(res_type[0].element_type) == "bf16":
        #     print("Running on BF16 mode, skipping numpy comparison.")
        # else:
        #     print(out)
        #     input1 = rt.ranked_memref_to_numpy(memref_ptrs[1][0])
        #     input2 = rt.ranked_memref_to_numpy(memref_ptrs[2][0])
        #     numpy_out = np.matmul(input1, input2)
        #     print(numpy_out)
        #     print(
        #         f"MLIR equal to NumPy? {np.allclose(out, numpy_out,rtol=1e-03, atol=1e-03)}"
        #     )


if __name__ == "__main__":
    test(
        "/workspaces/llvm/iree/Python/tmp.cubin",
        "/workspaces/llvm/iree/build/llvm-project",
    )
