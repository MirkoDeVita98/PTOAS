# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Case listing and verification for the swiglu_mx_quant kernel."""

from __future__ import annotations

from kernel_test.results import CaseResult

from .tile_config import DEFAULT_TILE, TOLERANCE


def _lightweight_cases() -> dict[str, object]:
    from .reference import _CASE_AXES
    from .tile_config import case_id

    return {
        case_id(dtype, outkind, rm, sa): {
            "dtype": dtype,
            "outkind": outkind,
            "round_mode": rm,
            "scalealg": sa,
            "tile": DEFAULT_TILE,
        }
        for dtype, outkind, rm, sa in _CASE_AXES
    }


def list_cases(workflow: str) -> dict[str, object]:
    """Return the swiglu_mx_quant case matrix for the requested workflow."""

    if workflow not in {"correctness", "cycle"}:
        raise ValueError(f"unsupported swiglu_mx_quant workflow: {workflow}")

    try:
        from .reference import generate_all
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy", "torch"}:
            raise
        return _lightweight_cases()

    return generate_all(tile=DEFAULT_TILE)


def verify_case(case_id: str, case: object, output: object) -> CaseResult:
    """Verify swiglu_mx_quant y (+ scale) bytes against the generated golden."""

    import numpy as np

    y_dev, scale_dev = output
    y_got = y_dev.cpu().numpy().astype(np.uint8).reshape(-1)
    scale_got = scale_dev.cpu().numpy().astype(np.uint8).reshape(-1)

    y_gold = np.asarray(case["y"], dtype=np.uint8).reshape(-1)
    scale_gold = np.asarray(case["scale"], dtype=np.uint8).reshape(-1)

    y_match = float((y_got == y_gold).mean()) if y_gold.size else 1.0
    scale_match = float((scale_got == scale_gold).mean()) if scale_gold.size else 1.0

    tol = TOLERANCE[case["outkind"]]
    tile = case["tile"]
    ok = y_match >= tol and scale_match >= tol
    message = (
        f"{case['dtype']}/{case['outkind']}/{case['round_mode']}/{case['scalealg']}: "
        f"y_match={y_match * 100:.2f}% scale_match={scale_match * 100:.2f}% "
        f"tile={tile.dim0}x{tile.dim1}"
    )
    return CaseResult(ok=ok, message=message)


def cycle_fields(case_id: str, case: object, backend: object) -> dict[str, object]:
    """Build stable cycle marker fields for one swiglu_mx_quant case."""

    del backend, case_id
    tile = case["tile"]
    return {
        "dtype": case["dtype"],
        "outkind": case["outkind"],
        "round": case["round_mode"],
        "scalealg": case["scalealg"],
        "dim0": tile.dim0,
        "dim1": tile.dim1,
    }
