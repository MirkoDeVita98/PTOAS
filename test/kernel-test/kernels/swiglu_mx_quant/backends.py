# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Backend factory for the swiglu_mx_quant kernel."""

from __future__ import annotations

from kernel_test.backends import BackendAdapter


def create_backend(name: str) -> BackendAdapter:
    """Create one swiglu_mx_quant backend adapter from the local backend packages."""

    if name == "cce":
        try:
            from .cce.backend import SwigluMxQuantCceBackend
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "swiglu_mx_quant cce backend requires the runtime dependencies; use the "
                "`pto` conda environment or install numpy/torch/torch_npu first"
            ) from exc

        return SwigluMxQuantCceBackend()
    if name == "vmi":
        try:
            from .vmi.backend import SwigluMxQuantVmiBackend
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "swiglu_mx_quant vmi backend requires the runtime dependencies; use the "
                "`pto` conda environment or install ptodsl/torch/torch_npu first"
            ) from exc

        return SwigluMxQuantVmiBackend()
    raise ValueError(f"unknown swiglu_mx_quant backend: {name}")
