# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""VMI (PTODSL) backend for swiglu_mx_quant (bf16 / e4m3 / OCP / swigluMode 0).

Two kernels, both processing rows/64 rows per core (work parity with CCE):

* Single-buffered (``_sb``): one row at a time, DMA-in -> compute -> DMA-out.
* Double-buffered (``_db``): rows processed in PAIRS (buf0 = even, buf1 = odd);
  the two computes run back-to-back on V while the MTE3 store of row0 overlaps the
  compute of row1 (store waits only on row0's compute flag, not blocking V). This
  mirrors CCE's DB scheme where MTE3 stores overlap vector compute — VMI's MTE3
  (store) span was the dominant cost at scale. The scalar surface has no runtime
  i%2, so buffers are compile-time; pairing gives DB with static buffer addresses.

Routing: DB is used when there are >=1 full pairs AND NTILE>=2 (a single-iteration
tile loop flattens and makes the two straight-line computes' CSE'd vregs escape their
VPTO vecscope). NTILE==1 shapes fall back to single-buffered (there VMI already beats
CCE, so DB is not needed). Per TILE-wide chunk: SwiGLU silu-gate -> vcmax(group=NBLK)
per-32-block max -> e8m0/recip bit math (E4M3 emax=8, OCP) -> vbrc(group=NBLK) recip
broadcast -> f8e4m3 quantize; scale bytes gathered one/block via stride-32 DMA.
"""

import os
from pathlib import Path

from ptodsl import pto, scalar as s

from kernel_test.backends import ArtifactPlan, RunPurpose
from kernel_test.npu_runtime import ensure_runtime, stream_ptr, sync

from ..runtime import SwigluMxQuantLaunchArgs, artifact_case_dir, prepare_launch_args

_VMI_ROOT = Path(__file__).resolve().parent
_GENERATED_DIR = _VMI_ROOT.parent / "generated"

_MAX_TILE = 256
_BLK = 32
_GRID = 64

_E4M3_EMAX = 8
_EXP_MASK_FP32 = 0x7F800000
_FP32_NAN_BITS = 0x7FC00000
_FP32_ONE_BITS = 0x3F800000


def _emit_tile_compute(gate_ptr, val_ptr, y_ptr, sw_ptr, TILE, NTILE, NBLK):
    """Emit the per-row SwiGLU + per-32-block MX-quant over NTILE 256-wide tiles."""
    with pto.for_(pto.const(0), pto.const(NTILE), step=1) as t:
        toff = s.muli(t, pto.const(TILE))
        mtile = pto.vmi.create_mask(TILE, size=TILE)
        mblk = pto.vmi.create_mask(NBLK, size=NBLK)
        x1 = pto.vmi.vcvt(pto.vmi.vload(gate_ptr, toff, size=TILE), pto.f32)
        x2 = pto.vmi.vcvt(pto.vmi.vload(val_ptr, toff, size=TILE), pto.f32)
        neg = pto.vmi.vmuls(x1, -1.0, mtile)
        e = pto.vmi.vexp(neg, mtile)
        d = pto.vmi.vadds(e, 1.0, mtile)
        sig = pto.vmi.vdiv(x1, d, mtile)
        out = pto.vmi.vmul(sig, x2, mtile)
        sw = pto.vmi.vcvt(pto.vmi.vcvt(out, pto.bf16, rounding="R"), pto.f32)

        gmax = pto.vmi.vcmax(pto.vmi.vabs(sw, mtile), mtile, group=NBLK)
        amax_bits = pto.vmi.vinterpret_cast(gmax, pto.ui32)
        exp = pto.vmi.vshrs(
            pto.vmi.vand(amax_bits, pto.vmi.vbrc(pto.ui32(_EXP_MASK_FP32), size=NBLK), mblk),
            23, mblk,
        )
        is_zero = pto.vmi.vcmps(amax_bits, 0, mblk, "eq")
        is_nan = pto.vmi.vcmps(exp, 0xFF, mblk, "eq")
        too_small = pto.vmi.vcmps(exp, _E4M3_EMAX, mblk, "lt")
        shared = pto.vmi.vsub(exp, pto.vmi.vbrc(pto.ui32(_E4M3_EMAX), size=NBLK), mblk)
        shared = pto.vmi.vsel(too_small, pto.vmi.vbrc(pto.ui32(0), size=NBLK), shared)
        e8 = pto.vmi.vsel(is_nan, pto.vmi.vbrc(pto.ui32(0xFF), size=NBLK), shared)
        e8 = pto.vmi.vsel(is_zero, pto.vmi.vbrc(pto.ui32(0), size=NBLK), e8)

        recip_bits = pto.vmi.vshls(
            pto.vmi.vsub(pto.vmi.vbrc(pto.ui32(254), size=NBLK), e8, mblk), 23, mblk,
        )
        recip = pto.vmi.vinterpret_cast(recip_bits, pto.f32)
        one = pto.vmi.vinterpret_cast(pto.vmi.vbrc(pto.ui32(_FP32_ONE_BITS), size=NBLK), pto.f32)
        nan = pto.vmi.vinterpret_cast(pto.vmi.vbrc(pto.ui32(_FP32_NAN_BITS), size=NBLK), pto.f32)
        recip = pto.vmi.vsel(is_zero, one, recip)
        recip = pto.vmi.vsel(is_nan, nan, recip)

        recip_tile = pto.vmi.vbrc(recip, size=TILE, group=NBLK)
        q = pto.vmi.vmul(sw, recip_tile, mtile)
        y = pto.vmi.vcvt(q, pto.f8e4m3, rounding="R", saturate="SAT")
        pto.vmi.vstore(y, y_ptr, toff, mtile)
        e8_tile = pto.vmi.vbrc(e8, size=TILE, group=NBLK)
        pto.vmi.vstore(pto.vmi.vcvt(e8_tile, pto.ui8), sw_ptr, toff, mtile)


@pto.jit(name="swiglu_mx_quant_vmi_sb", target="a5", backend="vpto",
         mode="explicit", kernel_kind="vector", insert_sync=False)
def swiglu_mx_quant_vmi_sb(
    x_gm: pto.ptr(pto.bf16, "gm"),
    y_gm: pto.ptr(pto.f8e4m3, "gm"),
    scale_gm: pto.ptr(pto.ui8, "gm"),
    *,
    DIM1: pto.const_expr = 512, HALF: pto.const_expr = 256, RPC: pto.const_expr = 1,
    TILE: pto.const_expr = 256, NTILE: pto.const_expr = 1, NBLK: pto.const_expr = 8,
):
    SR = HALF // _BLK
    gate_ptr = pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.bf16, "ub"))
    val_ptr = pto.castptr(pto.const(HALF * 2, dtype=pto.ui64), pto.ptr(pto.bf16, "ub"))
    y_ptr = pto.castptr(pto.const(HALF * 4, dtype=pto.ui64), pto.ptr(pto.f8e4m3, "ub"))
    sw_ptr = pto.castptr(pto.const(HALF * 5, dtype=pto.ui64), pto.ptr(pto.ui8, "ub"))
    base_row = s.muli(s.index_cast(pto.get_block_idx()), pto.const(RPC))
    with pto.for_(pto.const(0), pto.const(RPC), step=1) as i:
        row = s.addi(base_row, i)
        rx = s.muli(row, pto.const(DIM1))
        pto.mte_gm_ub(pto.addptr(x_gm, rx), gate_ptr, 0, HALF * 2, nburst=(1, HALF * 2, HALF * 2))
        pto.mte_gm_ub(pto.addptr(x_gm, s.addi(rx, pto.const(HALF))), val_ptr, 0, HALF * 2,
                      nburst=(1, HALF * 2, HALF * 2))
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
        _emit_tile_compute(gate_ptr, val_ptr, y_ptr, sw_ptr, TILE, NTILE, NBLK)
        pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=1)
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=1)
        pto.mte_ub_gm(y_ptr, pto.addptr(y_gm, s.muli(row, pto.const(HALF))), HALF,
                      nburst=(1, HALF, HALF))
        pto.mte_ub_gm(sw_ptr, pto.addptr(scale_gm, s.muli(row, pto.const(SR))), 1,
                      nburst=(SR, _BLK, 1))
        pto.set_flag(pto.Pipe.MTE3, pto.Pipe.MTE2, event_id=2)
        pto.wait_flag(pto.Pipe.MTE3, pto.Pipe.MTE2, event_id=2)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(name="swiglu_mx_quant_vmi_db", target="a5", backend="vpto",
         mode="explicit", kernel_kind="vector", insert_sync=False)
def swiglu_mx_quant_vmi_db(
    x_gm: pto.ptr(pto.bf16, "gm"),
    y_gm: pto.ptr(pto.f8e4m3, "gm"),
    scale_gm: pto.ptr(pto.ui8, "gm"),
    *,
    DIM1: pto.const_expr = 512, HALF: pto.const_expr = 256, RPC: pto.const_expr = 2,
    TILE: pto.const_expr = 256, NTILE: pto.const_expr = 2, NBLK: pto.const_expr = 8,
):
    SR = HALF // _BLK
    SET = HALF * 6  # bytes per buffer set

    def ptrs(base):
        return (pto.castptr(pto.const(base + 0, dtype=pto.ui64), pto.ptr(pto.bf16, "ub")),
                pto.castptr(pto.const(base + HALF * 2, dtype=pto.ui64), pto.ptr(pto.bf16, "ub")),
                pto.castptr(pto.const(base + HALF * 4, dtype=pto.ui64), pto.ptr(pto.f8e4m3, "ub")),
                pto.castptr(pto.const(base + HALF * 5, dtype=pto.ui64), pto.ptr(pto.ui8, "ub")))

    g0, v0, y0, sw0 = ptrs(0)
    g1, v1, y1, sw1 = ptrs(SET)

    def load(row, gp, vp):
        rx = s.muli(row, pto.const(DIM1))
        pto.mte_gm_ub(pto.addptr(x_gm, rx), gp, 0, HALF * 2, nburst=(1, HALF * 2, HALF * 2))
        pto.mte_gm_ub(pto.addptr(x_gm, s.addi(rx, pto.const(HALF))), vp, 0, HALF * 2,
                      nburst=(1, HALF * 2, HALF * 2))

    def store(row, yp, swp):
        pto.mte_ub_gm(yp, pto.addptr(y_gm, s.muli(row, pto.const(HALF))), HALF, nburst=(1, HALF, HALF))
        pto.mte_ub_gm(swp, pto.addptr(scale_gm, s.muli(row, pto.const(SR))), 1, nburst=(SR, _BLK, 1))

    base_row = s.muli(s.index_cast(pto.get_block_idx()), pto.const(RPC))
    with pto.for_(pto.const(0), pto.const(RPC // 2), step=1) as j:
        row0 = s.addi(base_row, s.muli(j, pto.const(2)))
        row1 = s.addi(row0, pto.const(1))
        load(row0, g0, v0)
        load(row1, g1, v1)
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
        _emit_tile_compute(g0, v0, y0, sw0, TILE, NTILE, NBLK)
        pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=2)   # row0 compute done
        _emit_tile_compute(g1, v1, y1, sw1, TILE, NTILE, NBLK)  # overlaps row0 store
        pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=3)   # row1 compute done
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=2)
        store(row0, y0, sw0)
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=3)
        store(row1, y1, sw1)
        pto.set_flag(pto.Pipe.MTE3, pto.Pipe.MTE2, event_id=4)
        pto.wait_flag(pto.Pipe.MTE3, pto.Pipe.MTE2, event_id=4)
    pto.pipe_barrier(pto.Pipe.ALL)


_COMPILED: dict[tuple, object] = {}


def _shape_params(case: dict) -> tuple:
    tile = case["tile"]
    dim1 = tile.dim1
    half = dim1 // 2
    rpc = tile.dim0 // _GRID
    tw = min(half, _MAX_TILE)
    ntile = half // tw
    nblk = tw // _BLK
    return (dim1, half, rpc, tw, ntile, nblk)


def _use_db(rpc, ntile):
    # Env override for A/B benchmarking: SMX_VMI_FORCE=sb|db forces the mode.
    force = os.environ.get("SMX_VMI_FORCE", "").lower()
    if force == "sb":
        return False
    if force == "db":
        return True
    return rpc >= 2 and ntile >= 2


def _prepare(case: dict) -> object:
    dim1, half, rpc, tw, ntile, nblk = _shape_params(case)
    db = _use_db(rpc, ntile)
    key = (dim1, half, rpc, tw, ntile, nblk, db)
    compiled = _COMPILED.get(key)
    if compiled is None:
        fn = swiglu_mx_quant_vmi_db if db else swiglu_mx_quant_vmi_sb
        compiled = fn.compile(DIM1=dim1, HALF=half, RPC=rpc, TILE=tw, NTILE=ntile, NBLK=nblk)
        _COMPILED[key] = compiled
    return compiled


def _launch(case: dict, launch_args: SwigluMxQuantLaunchArgs) -> object:
    compiled = _prepare(case)
    compiled[launch_args.block_dim, stream_ptr()](
        launch_args.x.data_ptr(),
        launch_args.y.data_ptr(),
        launch_args.mxscale.data_ptr(),
    )
    sync()
    return launch_args.y, launch_args.mxscale


class SwigluMxQuantVmiBackend:
    """Pure-PTODSL VMI backend for swiglu_mx_quant (single- or double-buffered)."""

    name = "vmi"

    def is_supported(self, case: object, *, purpose: RunPurpose) -> tuple[bool, str | None]:
        del purpose
        tile = case["tile"]
        half = tile.dim1 // 2
        ok = (
            case.get("dtype") == "bf16"
            and case.get("outkind") == "e4m3"
            and case.get("round_mode") == "rint"
            and case.get("scalealg") == "ocp"
            and tile.dim0 % _GRID == 0
            and half % _MAX_TILE == 0
        )
        if ok:
            return True, None
        return False, "vmi backend: bf16_e4m3_rint_ocp, rows%64==0, half%256==0"

    def launch(self, case: object, *, purpose: RunPurpose) -> object:
        ensure_runtime("swiglu_mx_quant")
        launch_args = prepare_launch_args(case, cycle=purpose == "cycle")
        return _launch(case, launch_args)

    def cache_tag(self) -> str:
        backend_py = _VMI_ROOT / "backend.py"
        return f"vmi:{backend_py}:{os.path.getmtime(backend_py):.0f}"

    def build_artifact_plan(self, case_id: str, case: object) -> ArtifactPlan:
        del case_id
        compiled = _prepare(case)
        case_dir = artifact_case_dir(_GENERATED_DIR, case, backend_name="vmi")
        return ArtifactPlan(
            generated_dir=_GENERATED_DIR,
            case_dir=case_dir,
            vmi_text=compiled.mlir_text(),
        )
