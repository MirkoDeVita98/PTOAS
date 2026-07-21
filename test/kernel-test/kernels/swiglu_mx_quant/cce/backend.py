# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CCE backend for the swiglu_mx_quant kernel."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

from kernel_test.backends import RunPurpose
from kernel_test.npu_runtime import ensure_runtime, stream_ptr, sync

from ..reference import _CASE_AXES
from ..runtime import SwigluMxQuantLaunchArgs, prepare_launch_args
from ..tile_config import sim_fn_name

_CCE_ROOT = Path(__file__).resolve().parent
_BUILD_DIR = _CCE_ROOT / "build"
_KERNEL_SOURCE = _CCE_ROOT / "swiglu_mx_quant_kernel.cpp"
_LIB_PATH = _BUILD_DIR / "libswiglu_mx_quant_cce.so"
_CMAKE_FILE = _CCE_ROOT / "CMakeLists.txt"

# Bind every extern-C launcher referenced by the generated case matrix.
_SIM_SYMBOLS: tuple[str, ...] = tuple(
    dict.fromkeys(
        sim_fn_name(dtype, outkind, rm, sa) for dtype, outkind, rm, sa in _CASE_AXES
    )
)
_LIB: ctypes.CDLL | None = None


def _ascend_home() -> Path:
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        raise EnvironmentError("ASCEND_HOME_PATH is not set. Source CANN setenv.bash first.")
    return Path(home)


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _build_lib(force: bool = False) -> Path:
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if _LIB_PATH.is_file() and not force:
        return _LIB_PATH

    ascend = _ascend_home()
    driver = os.environ.get("ASCEND_DRIVER_PATH", "/usr/local/Ascend/driver")
    _run(
        [
            "cmake",
            "-S",
            str(_CCE_ROOT),
            "-B",
            str(_BUILD_DIR),
            f"-DASCEND_HOME_PATH={ascend}",
            f"-DASCEND_DRIVER_PATH={driver}",
        ],
        cwd=_CCE_ROOT,
    )
    _run(["cmake", "--build", str(_BUILD_DIR), "--target", "swiglu_mx_quant_cce"], cwd=_CCE_ROOT)
    return _LIB_PATH


def _bind_lib(lib: ctypes.CDLL) -> None:
    argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_uint32]
    for name in _SIM_SYMBOLS:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(_build_lib()))
        _bind_lib(_LIB)
    return _LIB


def _vp(t) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _launch_cce(lib: ctypes.CDLL, launch_args: SwigluMxQuantLaunchArgs) -> object:
    fn = getattr(lib, launch_args.fn_name)
    fn(
        stream_ptr(),
        _vp(launch_args.x),
        _vp(launch_args.group_index),
        _vp(launch_args.y),
        _vp(launch_args.mxscale),
        _vp(launch_args.tiling),
        ctypes.c_uint32(launch_args.block_dim),
    )
    sync()
    return launch_args.y, launch_args.mxscale


class SwigluMxQuantCceBackend:
    """CCE swiglu_mx_quant backend implemented locally under kernel-test."""

    name = "cce"

    def is_supported(self, case: object, *, purpose: RunPurpose) -> tuple[bool, str | None]:
        del purpose
        if case.get("outkind") not in ("e4m3", "e5m2"):
            return False, "cce backend milestone 1 verifies fp8 (e4m3/e5m2) OCP only"
        return True, None

    def launch(self, case: object, *, purpose: RunPurpose) -> object:
        ensure_runtime("swiglu_mx_quant")
        launch_args = prepare_launch_args(case, cycle=purpose == "cycle")
        return _launch_cce(_load_lib(), launch_args)
