# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU golden references for the swiglu_mx_quant kernel (SwiGLU + per-32-block MX quant).

Golden semantics for the primary path (swigluMode 0, activateLeft 0, OCP scale):
  - Split the last dim in half: x1 = first half, x2 = second half.
  - gate = silu(x1) = x1 * sigmoid(x1) = x1 / (1 + exp(-x1)) (computed in f32).
  - swiglu = round_to_input_dtype(gate * x2).
  - Per 32-element MX block: shared_exp = floor(log2(amax)) - element_emax,
    e8m0 scale byte = clamp(shared_exp + 127, 0, 255), scale = 2**shared_exp.
  - y = to_fp8(swiglu / scale).
The MX-quant math is adapted from vmi-demo/quant/per_block_cast/cce/ref/golden.py.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import torch

from .tile_config import (
    BLOCK_SIZE,
    DEFAULT_TILE,
    ELEMENT_EMAX,
    TileConfig,
    case_id,
)

SEED = 42

# --- CCE ABI constants (mirror common/tiling.py) ---
VECTOR_CORE_NUM = 64
UB_SIZE = 262144
BYTES_OF_FP16 = 2
BYTES_OF_INT16 = 2
BYTES_OF_FP8 = 1
RESERVED_UB_SIZE = 32
RESERVED_UB_FOR_ALIGN = 128
DOUBLE_BUFFER = 2
CONST_TWO = 2
X_ONCE_NUM = 512
QUANT_ONCE_NUM = 256
SCALE_ONCE_NUM = 8
BASE_LAST_FACTOR_DIM1 = 256

DTYPE_E5M2 = 35
DTYPE_E4M3 = 36
DTYPE_E2M1 = 40
DTYPE_E1M2 = 41

TPL_RINT = 1
TPL_ROUND = 0
TPL_FLOOR = 4
TPL_SCALE_ALG_0 = 0
TPL_SCALE_ALG_1 = 1

_OUTKIND_TO_DST = {
    "e4m3": DTYPE_E4M3,
    "e5m2": DTYPE_E5M2,
    "e2m1": DTYPE_E2M1,
    "e1m2": DTYPE_E1M2,
}
_ROUND_TO_VAL = {"rint": TPL_RINT, "round": TPL_ROUND, "floor": TPL_FLOOR}
_SCALEALG_TO_VAL = {"ocp": TPL_SCALE_ALG_0, "cublas": TPL_SCALE_ALG_1}

# Cases generated for correctness/cycle. Milestone 1 exercises the fp8 OCP paths.
_CASE_AXES = [
    ("bf16", "e4m3", "rint", "ocp"),
    ("f16", "e4m3", "rint", "ocp"),
    ("bf16", "e5m2", "rint", "ocp"),
    ("f16", "e5m2", "rint", "ocp"),
]

TILING_FORMAT = "<24q4f"


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def floor_div(a: int, b: int) -> int:
    return a // b


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "f16":
        return torch.float16
    raise ValueError(f"unknown dtype: {dtype}")


def _fp8_dtype(outkind: str) -> torch.dtype:
    if outkind == "e4m3":
        return torch.float8_e4m3fn
    if outkind == "e5m2":
        return torch.float8_e5m2
    raise ValueError(f"reference only implements fp8 out kinds, got {outkind}")


def _fp8_clamp(outkind: str) -> float:
    # Max finite magnitude for each fp8 kind.
    return 448.0 if outkind == "e4m3" else 57344.0


# --------------------------------------------------------------------------- #
# Tiling (ported from cce/swiglu_mx_quant/common/tiling.py calc_tiling)
# --------------------------------------------------------------------------- #
def compute_tiling_bytes(
    input_shape: tuple[int, int],
    outkind: str,
    round_mode: str,
    scalealg: str,
    *,
    activate_left: bool = False,
    swiglu_mode: int = 0,
    clamp_limit: float = 7.0,
    glu_alpha: float = 1.702,
    glu_bias: float = 1.0,
) -> tuple[bytes, int]:
    """Return (serialized_tiling_bytes, used_core_num). Mirrors CalcTiling."""

    ge_dst = _OUTKIND_TO_DST[outkind]
    rm_val = _ROUND_TO_VAL[round_mode]
    sa_val = _SCALEALG_TO_VAL[scalealg]

    dim_num = len(input_shape)
    input_dim1 = 1
    for i in range(dim_num - 1):
        input_dim1 *= input_shape[i]
    input_dim2 = input_shape[dim_num - 1]

    fields = {
        "usedCoreNum": 0,
        "inputDim1": input_dim1,
        "inputDim2": input_dim2,
        "outputDim2": input_dim2 // CONST_TWO,
        "basicDim2": BASE_LAST_FACTOR_DIM1,
        "basicDim1": 1,
        "maxBasicNumUbDim2": 0,
        "maxBasicNumUbDim1": 0,
        "ubLoopPerRow": 0,
        "ubTailPerRow": 0,
        "frontCoreNum": 0,
        "frontCoreBasicNumDim1": 0,
        "frontCoreLoopTimes": 0,
        "frontCoreLastLoopBasicNum": 0,
        "tailCoreBasicNumDim1": 0,
        "tailCoreLoopTimes": 0,
        "tailCoreLastLoopBasicNum": 0,
        "activateLeft": 1 if activate_left else 0,
        "swigluMode": swiglu_mode,
        "roundMode": rm_val,
        "scaleAlg": sa_val,
        "groupMode": 0,
        "groupIndexNum": 0,
        "useDoubleBuffer": 0,
        "clampLimit": float(clamp_limit),
        "gluAlpha": float(glu_alpha),
        "gluBias": float(glu_bias),
        "maxDtypeValue": 0.0,
    }

    basic_dim1 = fields["basicDim1"]
    basic_dim2 = fields["basicDim2"]
    output_dim2 = fields["outputDim2"]
    is_fp4 = ge_dst in (DTYPE_E2M1, DTYPE_E1M2)

    available_ub = UB_SIZE - RESERVED_UB_SIZE - RESERVED_UB_FOR_ALIGN

    bytes_per_iteration = 0
    bytes_per_iteration += basic_dim1 * basic_dim2 * CONST_TWO * BYTES_OF_FP16
    if is_fp4:
        bytes_per_iteration += basic_dim1 * basic_dim2 // CONST_TWO
    else:
        bytes_per_iteration += basic_dim1 * basic_dim2 * BYTES_OF_FP8
    bytes_per_iteration += basic_dim1 * basic_dim2 // BLOCK_SIZE * BYTES_OF_FP8
    bytes_per_iteration *= DOUBLE_BUFFER
    bytes_per_iteration += basic_dim1 * basic_dim2 // BLOCK_SIZE * BYTES_OF_FP16
    bytes_per_iteration += basic_dim1 * basic_dim2 // BLOCK_SIZE * BYTES_OF_INT16
    bytes_per_iteration += basic_dim1 * basic_dim2 * BYTES_OF_FP16

    ub_total_basic_block = available_ub // bytes_per_iteration
    basic_per_row = ceil_div(output_dim2, BASE_LAST_FACTOR_DIM1)

    if ub_total_basic_block >= basic_per_row:
        fields["maxBasicNumUbDim2"] = basic_per_row
        fields["maxBasicNumUbDim1"] = floor_div(ub_total_basic_block, basic_per_row)
        fields["ubLoopPerRow"] = 1
        fields["ubTailPerRow"] = output_dim2
    else:
        fields["maxBasicNumUbDim2"] = ub_total_basic_block
        fields["maxBasicNumUbDim1"] = 1
        fields["ubLoopPerRow"] = ceil_div(basic_per_row, fields["maxBasicNumUbDim2"])
        fields["ubTailPerRow"] = output_dim2 - (fields["ubLoopPerRow"] - 1) * fields["maxBasicNumUbDim2"] * basic_dim2

    basic_per_col = ceil_div(input_dim1, basic_dim1)
    fields["usedCoreNum"] = min(basic_per_col, VECTOR_CORE_NUM)

    tail_core_basic_num_dim1 = floor_div(basic_per_col, fields["usedCoreNum"])
    tail_core_loop_times = ceil_div(tail_core_basic_num_dim1, fields["maxBasicNumUbDim1"])
    tail_core_last_loop_basic_num = (
        tail_core_basic_num_dim1 - (tail_core_loop_times - 1) * fields["maxBasicNumUbDim1"]
    )

    fields["frontCoreNum"] = basic_per_col % fields["usedCoreNum"]
    fields["frontCoreBasicNumDim1"] = tail_core_basic_num_dim1 + 1
    fields["frontCoreLoopTimes"] = ceil_div(fields["frontCoreBasicNumDim1"], fields["maxBasicNumUbDim1"])
    fields["frontCoreLastLoopBasicNum"] = (
        fields["frontCoreBasicNumDim1"] - (fields["frontCoreLoopTimes"] - 1) * fields["maxBasicNumUbDim1"]
    )
    fields["tailCoreBasicNumDim1"] = tail_core_basic_num_dim1
    fields["tailCoreLoopTimes"] = tail_core_loop_times
    fields["tailCoreLastLoopBasicNum"] = tail_core_last_loop_basic_num

    # Double-buffer UB budget check (shared-compute-buffers layout).
    fs = fields["maxBasicNumUbDim1"] * fields["maxBasicNumUbDim2"]

    def _ub_budget(f: int) -> int:
        input_per_tensor = f * (X_ONCE_NUM // CONST_TWO) * BYTES_OF_FP16
        db_per_set = 2 * input_per_tensor + ((f * SCALE_ONCE_NUM * BYTES_OF_FP8 + 63) // 64) * 64
        if is_fp4:
            db_per_set += ((f * QUANT_ONCE_NUM // CONST_TWO + 31) // 32) * 32
        else:
            db_per_set += ((f * QUANT_ONCE_NUM + 31) // 32) * 32
        compute_region = (
            f * QUANT_ONCE_NUM * BYTES_OF_FP16
            + ((f * SCALE_ONCE_NUM * BYTES_OF_INT16 + 31) // 32) * 32 * 2
        )
        return 2 * db_per_set + compute_region

    if _ub_budget(fs) <= available_ub:
        fields["useDoubleBuffer"] = 1
    else:
        best_fs = 1
        for candidate_fs in range(fs - 1, 0, -1):
            if _ub_budget(candidate_fs) <= available_ub:
                best_fs = candidate_fs
                break
        fields["useDoubleBuffer"] = 1
        eff_fs = best_fs
        if basic_per_row > 0:
            fields["maxBasicNumUbDim2"] = min(basic_per_row, eff_fs)
        else:
            fields["maxBasicNumUbDim2"] = 1
        if fields["maxBasicNumUbDim2"] > 0:
            fields["maxBasicNumUbDim1"] = max(1, eff_fs // fields["maxBasicNumUbDim2"])
        else:
            fields["maxBasicNumUbDim1"] = 1
        if fields["maxBasicNumUbDim2"] >= basic_per_row:
            fields["ubLoopPerRow"] = 1
            fields["ubTailPerRow"] = output_dim2
        else:
            fields["ubLoopPerRow"] = ceil_div(basic_per_row, fields["maxBasicNumUbDim2"])
            fields["ubTailPerRow"] = (
                output_dim2 - (fields["ubLoopPerRow"] - 1) * fields["maxBasicNumUbDim2"] * basic_dim2
            )
        tail_core_basic_num_dim1 = floor_div(basic_per_col, fields["usedCoreNum"])
        tail_core_loop_times = ceil_div(tail_core_basic_num_dim1, fields["maxBasicNumUbDim1"])
        tail_core_last_loop_basic_num = (
            tail_core_basic_num_dim1 - (tail_core_loop_times - 1) * fields["maxBasicNumUbDim1"]
        )
        fields["frontCoreNum"] = basic_per_col % fields["usedCoreNum"]
        fields["frontCoreBasicNumDim1"] = tail_core_basic_num_dim1 + 1
        fields["frontCoreLoopTimes"] = ceil_div(fields["frontCoreBasicNumDim1"], fields["maxBasicNumUbDim1"])
        fields["frontCoreLastLoopBasicNum"] = (
            fields["frontCoreBasicNumDim1"] - (fields["frontCoreLoopTimes"] - 1) * fields["maxBasicNumUbDim1"]
        )
        fields["tailCoreBasicNumDim1"] = tail_core_basic_num_dim1
        fields["tailCoreLoopTimes"] = tail_core_loop_times
        fields["tailCoreLastLoopBasicNum"] = tail_core_last_loop_basic_num

    packed = struct.pack(
        TILING_FORMAT,
        fields["usedCoreNum"],
        fields["inputDim1"],
        fields["inputDim2"],
        fields["outputDim2"],
        fields["basicDim2"],
        fields["basicDim1"],
        fields["maxBasicNumUbDim2"],
        fields["maxBasicNumUbDim1"],
        fields["ubLoopPerRow"],
        fields["ubTailPerRow"],
        fields["frontCoreNum"],
        fields["frontCoreBasicNumDim1"],
        fields["frontCoreLoopTimes"],
        fields["frontCoreLastLoopBasicNum"],
        fields["tailCoreBasicNumDim1"],
        fields["tailCoreLoopTimes"],
        fields["tailCoreLastLoopBasicNum"],
        fields["activateLeft"],
        fields["swigluMode"],
        fields["roundMode"],
        fields["scaleAlg"],
        fields["groupMode"],
        fields["groupIndexNum"],
        fields["useDoubleBuffer"],
        fields["clampLimit"],
        fields["gluAlpha"],
        fields["gluBias"],
        fields["maxDtypeValue"],
    )
    return packed, fields["usedCoreNum"]


# --------------------------------------------------------------------------- #
# Golden compute
# --------------------------------------------------------------------------- #
def _make_input(tile: TileConfig, dtype: str) -> torch.Tensor:
    """Smooth, deterministic input. Linear ramps give near-bitwise CCE match."""

    torch.manual_seed(SEED)
    n = tile.dim0 * tile.dim1
    ramp = torch.linspace(-3.0, 3.0, steps=n, dtype=torch.float32).reshape(tile.x_shape)
    return ramp.to(_torch_dtype(dtype))


def _swiglu_mode0(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU mode 0, activateLeft 0: gate=silu(first half), out=gate*second half."""

    half = x.shape[-1] // 2
    x1 = x[..., :half].float()  # activated half
    x2 = x[..., half:].float()  # multiplied half
    sig = x1 / (1.0 + torch.exp(-x1))
    return sig * x2


def _mx_quant_fp8(swiglu: torch.Tensor, outkind: str, input_dtype: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-32-block MX quant of the (already input-dtype-rounded) swiglu tensor."""

    emax = ELEMENT_EMAX[outkind]
    dim0, half_input = swiglu.shape
    nblocks = half_input // BLOCK_SIZE
    blocks = swiglu.reshape(dim0, nblocks, BLOCK_SIZE).float()
    amax = blocks.abs().amax(dim=-1)  # (dim0, nblocks)

    e8m0 = np.zeros((dim0, nblocks), dtype=np.uint8)
    scale = np.ones((dim0, nblocks), dtype=np.float32)
    amax_np = amax.cpu().numpy()
    for r in range(dim0):
        for b in range(nblocks):
            a = float(amax_np[r, b])
            if a == 0.0:
                e8m0[r, b] = 0
                scale[r, b] = 1.0
            elif not math.isfinite(a):
                e8m0[r, b] = 255
                scale[r, b] = 1.0
            else:
                shared_exp = math.floor(math.log2(a)) - emax
                e8 = max(0, min(255, shared_exp + 127))
                e8m0[r, b] = np.uint8(e8)
                scale[r, b] = float(2.0 ** (e8 - 127))

    scale_t = torch.from_numpy(scale).reshape(dim0, nblocks, 1)
    scaled = (blocks / scale_t).reshape(dim0, half_input)
    clamp = _fp8_clamp(outkind)
    scaled = scaled.clamp(-clamp, clamp)
    y = scaled.to(_fp8_dtype(outkind)).view(torch.uint8).cpu().numpy().reshape(dim0, half_input)

    # Scale bytes must be padded to scale_bytes_per_row (2-byte aligned) per row.
    per_row = (half_input + BLOCK_SIZE - 1) // BLOCK_SIZE
    if per_row % 2 != 0:
        per_row += 1
    scale_out = np.zeros((dim0, per_row), dtype=np.uint8)
    scale_out[:, :nblocks] = e8m0
    return y, scale_out


def generate_case(
    dtype: str,
    outkind: str,
    round_mode: str,
    scalealg: str,
    tile: TileConfig | None = None,
) -> dict:
    tile = tile or DEFAULT_TILE
    if outkind not in ("e4m3", "e5m2"):
        raise ValueError(f"reference golden only implements fp8 kinds, got {outkind}")

    x = _make_input(tile, dtype)
    swiglu = _swiglu_mode0(x).to(_torch_dtype(dtype))  # narrow to input dtype like the kernel
    y_golden, scale_golden = _mx_quant_fp8(swiglu, outkind, dtype)

    tiling_bytes, used_core_num = compute_tiling_bytes(tile.x_shape, outkind, round_mode, scalealg)

    if dtype == "bf16":
        x_np = x.float().numpy()
    else:
        x_np = x.numpy()

    return {
        "dtype": dtype,
        "outkind": outkind,
        "round_mode": round_mode,
        "scalealg": scalealg,
        "tile": tile,
        "x": x_np,
        "y": y_golden,
        "scale": scale_golden,
        "tiling": tiling_bytes,
        "blockDim": used_core_num,
    }


def generate_all(tile: TileConfig | None = None) -> dict[str, dict]:
    tile = tile or DEFAULT_TILE
    cases: dict[str, dict] = {}
    for dtype, outkind, round_mode, scalealg in _CASE_AXES:
        cid = case_id(dtype, outkind, round_mode, scalealg)
        cases[cid] = generate_case(dtype, outkind, round_mode, scalealg, tile=tile)
    # Shape sweep for the primary path (bf16 e4m3 rint ocp), for CCE-vs-VMI parity.
    from .tile_config import SWEEP_SHAPES
    for (d0, d1) in SWEEP_SHAPES:
        st = TileConfig(f"{d0}x{d1}", dim0=d0, dim1=d1)
        cases[f"bf16_e4m3_rint_ocp_{d0}x{d1}"] = generate_case(
            "bf16", "e4m3", "rint", "ocp", tile=st
        )
    return cases
