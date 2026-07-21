# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared tile / case configuration for the swiglu_mx_quant kernel."""

from __future__ import annotations

from dataclasses import dataclass

# Input dtypes supported by the CCE kernel ABI.
DTYPES = ("bf16", "f16")

# Quantized output kinds (fp8 / fp4). Milestone-1 focuses on e4m3.
OUT_KINDS = ("e4m3", "e5m2", "e2m1", "e1m2")

# Rounding modes and scale algorithms exposed by the extern-C suffixes.
ROUND_MODES = ("rint", "round", "floor")
SCALE_ALGS = ("ocp", "cublas")

# MX block size: one e8m0 scale per 32 elements.
BLOCK_SIZE = 32

# Element emax used by the OCP shared-exponent derivation.
ELEMENT_EMAX = {
    "e4m3": 8,
    "e5m2": 15,
    "e2m1": 2,
    "e1m2": 0,
}

# Byte-match tolerance (fraction of bytes that must match golden) per output kind.
TOLERANCE = {
    "e4m3": 0.99,
    "e5m2": 0.99,
    "e2m1": 0.99,
    "e1m2": 0.99,
}

DEFAULT_DIM0 = 64
DEFAULT_DIM1 = 512


@dataclass(frozen=True)
class TileConfig:
    name: str
    dim0: int
    dim1: int

    @property
    def x_shape(self) -> tuple[int, int]:
        return (self.dim0, self.dim1)

    @property
    def half_input(self) -> int:
        return self.dim1 // 2

    @property
    def y_shape(self) -> tuple[int, int]:
        return (self.dim0, self.half_input)

    @property
    def scale_bytes_per_row(self) -> int:
        raw = (self.half_input + BLOCK_SIZE - 1) // BLOCK_SIZE
        if raw % 2 != 0:
            raw += 1
        return raw

    @property
    def scale_shape(self) -> tuple[int, int]:
        return (self.dim0, self.scale_bytes_per_row)


DEFAULT_TILE = TileConfig("prod", dim0=DEFAULT_DIM0, dim1=DEFAULT_DIM1)


def sim_fn_name(dtype: str, outkind: str, round_mode: str, scalealg: str) -> str:
    """Return the extern-C launcher symbol for the requested configuration."""

    return f"call_swiglu_mx_quant_{dtype}_{outkind}_{round_mode}_{scalealg}"


def case_id(dtype: str, outkind: str, round_mode: str, scalealg: str) -> str:
    return f"{dtype}_{outkind}_{round_mode}_{scalealg}"


# Shape sweep for CCE-vs-VMI equal-work benchmarking (rows divisible by 64).
SWEEP_SHAPES = ((64, 512), (128, 256), (128, 512), (256, 512), (512, 1024), (1024, 2048))
