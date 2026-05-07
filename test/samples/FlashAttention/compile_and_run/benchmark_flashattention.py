# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ctypes
import math

import torch
import torch_npu  # noqa: F401

KERNELS = [
    ("fa_140tflops", "/tmp/fa_140tflops.so", 524288, True),
    ("patched", "/tmp/compiler_team_fa.so", 229376, False),
]
DEVICE = "npu:0"
WARMUP_ITERS = 10
BENCH_ITERS = 100
NUM_CUBE_CORES = 24
RTOL = 1e-3
ATOL = 1e-3

Q_ROWS = 3072
HEAD = 128
S1_TOTAL = 8192
NUM_Q_BLOCKS = Q_ROWS // 32


def load_lib(lib_path, pass_shape):
    lib = ctypes.CDLL(lib_path)
    argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    if pass_shape:
        argtypes += [ctypes.c_int64, ctypes.c_int64]
    lib.call_kernel.argtypes = argtypes
    lib.call_kernel.restype = None
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def fused_attention(q_bsh, k_bsh, v_bsh):
    scale = 1.0 / math.sqrt(q_bsh.shape[-1])
    out, _ = torch_npu.npu_fused_infer_attention_score(
        q_bsh,
        k_bsh,
        v_bsh,
        num_heads=1,
        input_layout="BSH",
        scale=scale,
        next_tokens=65535,
    )
    return out


def fa_reference(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[1])
    scores = q.float() @ k.float().T * scale
    return torch.softmax(scores, dim=-1) @ v.float()


def run_pto_kernel(lib, pass_shape, block_dim, gm, q, k, v, o):
    stream = torch.npu.current_stream()._as_parameter_
    args = [block_dim, stream, ptr(gm), ptr(q), ptr(k), ptr(v), ptr(o)]
    if pass_shape:
        args += [q.shape[0], k.shape[0]]
    lib.call_kernel(*args)


def check_close(out_pto, out_fp32, out_torch_npu):
    max_err_fp32 = (out_pto - out_fp32).abs().max().item()
    max_err_torch_npu = (out_pto - out_torch_npu).abs().max().item()
    try:
        torch.testing.assert_close(out_pto, out_fp32, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(out_pto, out_torch_npu, rtol=RTOL, atol=ATOL)
        return "PASSED", max_err_fp32, max_err_torch_npu
    except AssertionError:
        return "FAILED", max_err_fp32, max_err_torch_npu


def bench(fn):
    for _ in range(WARMUP_ITERS):
        fn()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(BENCH_ITERS):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / BENCH_ITERS


def main():
    device = torch.device(DEVICE)
    block_dim = min(NUM_Q_BLOCKS, NUM_CUBE_CORES)
    flops = 4 * Q_ROWS * HEAD * S1_TOTAL

    torch.manual_seed(0)
    q = torch.randn((Q_ROWS, HEAD), dtype=torch.float16, device=device)
    k = torch.randn((S1_TOTAL, HEAD), dtype=torch.float16, device=device)
    v = torch.randn((S1_TOTAL, HEAD), dtype=torch.float16, device=device)
    q_bsh = q.unsqueeze(0)
    k_bsh = k.unsqueeze(0)
    v_bsh = v.unsqueeze(0)

    def run_torch_npu():
        fused_attention(q_bsh, k_bsh, v_bsh)

    out_torch_npu = fused_attention(q_bsh, k_bsh, v_bsh).squeeze(0).float().cpu()
    out_fp32 = fa_reference(q, k, v).float().cpu()
    torch.npu.synchronize()

    torch_npu_us = bench(run_torch_npu)
    torch_npu_tflops = flops / (torch_npu_us * 1e-6) / 1e12

    print(
        f"PTO FA variants vs torch_npu fused attention: Q={Q_ROWS} S1={S1_TOTAL} H={HEAD} "
        f"blockDim={block_dim}"
    )
    print(f"  torch_npu: {torch_npu_us:8.2f} us  {torch_npu_tflops:7.3f} TFLOP/s")

    for name, lib_path, gm_elems_per_block, pass_shape in KERNELS:
        lib = load_lib(lib_path, pass_shape)
        gm = torch.zeros(
            (gm_elems_per_block * block_dim,), dtype=torch.float32, device=device
        )
        o = torch.zeros((Q_ROWS, HEAD), dtype=torch.float32, device=device)

        def run_pto():
            run_pto_kernel(lib, pass_shape, block_dim, gm, q, k, v, o)

        # Correctness check against torch_npu fused attention.
        gm.zero_()
        o.zero_()
        run_pto()
        torch.npu.synchronize()
        out_pto = o.float().cpu()
        correctness, max_err_fp32, max_err_torch_npu = check_close(
            out_pto, out_fp32, out_torch_npu
        )

        pto_us = bench(run_pto)
        pto_tflops = flops / (pto_us * 1e-6) / 1e12
        print(
            f"  {name:12s}: {pto_us:8.2f} us  {pto_tflops:7.3f} TFLOP/s  "
            f"speedup={torch_npu_us / pto_us:.2f}x  {correctness}  "
            f"max_err(fp32={max_err_fp32:.3e}, torch_npu={max_err_torch_npu:.3e})"
        )


if __name__ == "__main__":
    main()
