# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Runtime data preparation for the swiglu_mx_quant kernel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kernel_test.npu_runtime import device_str, empty_npu

from .tile_config import sim_fn_name

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class SwigluMxQuantLaunchArgs:
    """Prepared runtime arguments shared by all swiglu_mx_quant backends."""

    dtype: str
    outkind: str
    round_mode: str
    scalealg: str
    fn_name: str
    x: "torch.Tensor"
    group_index: "torch.Tensor"
    y: "torch.Tensor"
    mxscale: "torch.Tensor"
    tiling: "torch.Tensor"
    block_dim: int


def torch_dtype(dtype: str) -> "torch.dtype":
    import torch

    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "f16":
        return torch.float16
    raise ValueError(f"unknown dtype: {dtype}")


def artifact_case_dir_name(case: dict) -> str:
    tile = case["tile"]
    return f"{case['dtype']}_{case['outkind']}_{case['round_mode']}_{case['scalealg']}_{tile.dim0}x{tile.dim1}"


def artifact_case_dir(root: Path, case: dict, *, backend_name: str) -> Path:
    return root / backend_name / artifact_case_dir_name(case)


def prepare_launch_args(case: dict, *, cycle: bool = False) -> SwigluMxQuantLaunchArgs:
    """Convert one case into prepared device tensors and launch metadata."""

    import numpy as np
    import torch

    del cycle  # symbol name does not depend on the workflow for this kernel

    dtype = case["dtype"]
    outkind = case["outkind"]
    round_mode = case["round_mode"]
    scalealg = case["scalealg"]
    dev = device_str()

    x = torch.from_numpy(np.ascontiguousarray(case["x"])).to(torch_dtype(dtype)).to(dev)

    tile = case["tile"]
    y = empty_npu(tile.y_shape, torch.uint8)
    mxscale = empty_npu(tile.scale_shape, torch.uint8)

    # groupMode=0: group_index is unused by the kernel but bind a small zero buffer.
    # Build via numpy->copy to avoid dispatching to aclnn factory ops.
    group_index = torch.from_numpy(np.zeros((32,), dtype=np.int32)).to(dev)

    tiling_np = np.frombuffer(bytearray(case["tiling"]), dtype=np.uint8).copy()
    tiling = torch.from_numpy(tiling_np).to(dev)

    return SwigluMxQuantLaunchArgs(
        dtype=dtype,
        outkind=outkind,
        round_mode=round_mode,
        scalealg=scalealg,
        fn_name=sim_fn_name(dtype, outkind, round_mode, scalealg),
        x=x,
        group_index=group_index,
        y=y,
        mxscale=mxscale,
        tiling=tiling,
        block_dim=int(case["blockDim"]),
    )
