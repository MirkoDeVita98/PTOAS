"""PTODSL Python-frontend backend for rope VF sim tests.

Unlike ``pto.backend`` (the ``vmi`` backend), the kernels below are traced
from real Python function bodies via ``ptodsl`` -- they are not loaded from
a static ``.pto`` text file. They reproduce the hardware-exposed "MI"
reference kernels in ``rope_{f16,bf16,f32}.mi.pto``.
"""

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(__file__).resolve().parents[3]


def _candidate_ptodsl_roots() -> list[Path]:
    candidates: list[Path] = []

    env_keys = ("PTODSL_PKG_ROOT", "PTOAS_ROOT", "PTOAS_HOME")
    for key in env_keys:
        raw = os.environ.get(key)
        if not raw:
            continue
        base = Path(raw).expanduser()
        if key == "PTODSL_PKG_ROOT":
            candidates.append(base)
        else:
            candidates.append(base / "ptodsl")

    candidates.append(REPO_ROOT / "PTOAS" / "ptodsl")
    for sibling in sorted(REPO_ROOT.parent.glob("PTOAS*")):
        candidates.append(sibling / "ptodsl")

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _import_ptodsl_pto():
    try:
        return importlib.import_module("ptodsl").pto
    except ModuleNotFoundError as first_error:
        for pkg_root in _candidate_ptodsl_roots():
            if not (pkg_root / "ptodsl" / "__init__.py").exists():
                continue
            pkg_root_str = str(pkg_root)
            if pkg_root_str not in sys.path:
                sys.path.insert(0, pkg_root_str)
            try:
                return importlib.import_module("ptodsl").pto
            except ModuleNotFoundError:
                continue
        raise ModuleNotFoundError(
            "Unable to import ptodsl. Set PTODSL_PKG_ROOT or PTOAS_ROOT, "
            "or place a PTOAS checkout next to this repo."
        ) from first_error


pto = _import_ptodsl_pto()


def _dma_round_up_32(nbytes):
    return ((nbytes + 31) // 32) * 32


@pto.jit(name="rope_mi_f32", target="a5", kernel_kind="vector", mode="explicit", insert_sync=False)
def rope_mi_f32(
    x_ptr: pto.ptr(pto.ui32, "gm"),
    cos_ptr: pto.ptr(pto.ui32, "gm"),
    sin_ptr: pto.ptr(pto.ui32, "gm"),
    y_ptr: pto.ptr(pto.ui32, "gm"),
    s_count: pto.i32,
    n_count: pto.i32,
    mode: pto.i32,
):
    gm_f32 = pto.ptr(pto.f32, "gm")
    ub_f32 = pto.ptr(pto.f32, "ub")
    x_gm = pto.castptr(x_ptr, gm_f32)
    cos_gm = pto.castptr(cos_ptr, gm_f32)
    sin_gm = pto.castptr(sin_ptr, gm_f32)
    y_gm = pto.castptr(y_ptr, gm_f32)

    cos_dma_bytes = _dma_round_up_32(s_count * 64 * 4)
    xy_dma_bytes = _dma_round_up_32(s_count * n_count * 64 * 4)

    cos_ub = pto.castptr(pto.const(0, dtype=pto.i64), ub_f32)
    sin_ub = pto.castptr(cos_dma_bytes, ub_f32)
    x_ub = pto.castptr(cos_dma_bytes + cos_dma_bytes, ub_f32)
    y_ub = pto.castptr(cos_dma_bytes + cos_dma_bytes + xy_dma_bytes, ub_f32)

    pto.mte_load(cos_gm, cos_ub, 0, cos_dma_bytes, nburst=(1, cos_dma_bytes, cos_dma_bytes))
    pto.mte_load(sin_gm, sin_ub, 0, cos_dma_bytes, nburst=(1, cos_dma_bytes, cos_dma_bytes))
    pto.mte_load(x_gm, x_ub, 0, xy_dma_bytes, nburst=(1, xy_dma_bytes, xy_dma_bytes))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    is_half_mode = mode == 0
    vf32 = pto.vreg_type(64, pto.f32)

    if is_half_mode:
        with pto.simd():
            mask = pto.pge_b32("PAT_VL32")
            for s in range(0, s_count, 1):
                cs_off = s * 64
                x_s_off = s * n_count * 64
                cos_lo = pto.vlds(cos_ub, cs_off, vf32)
                cos_hi = pto.vlds(cos_ub, cs_off + 32, vf32)
                sin_lo = pto.vlds(sin_ub, cs_off, vf32)
                sin_hi = pto.vlds(sin_ub, cs_off + 32, vf32)
                for n in range(0, n_count, 1):
                    x_off = x_s_off + n * 64
                    x_lo = pto.vlds(x_ub, x_off, vf32)
                    x_hi = pto.vlds(x_ub, x_off + 32, vf32)
                    y_lo = pto.vsub(pto.vmul(cos_lo, x_lo, mask), pto.vmul(sin_lo, x_hi, mask), mask)
                    y_hi = pto.vadd(pto.vmul(cos_hi, x_hi, mask), pto.vmul(sin_hi, x_lo, mask), mask)
                    pto.vsts(y_lo, y_ub, x_off, mask)
                    pto.vsts(y_hi, y_ub, x_off + 32, mask)
    else:
        with pto.simd():
            mask_pair = pto.pge_b32("PAT_VL32")
            mask_full = pto.pset_b32("PAT_ALL")
            for s in range(0, s_count, 1):
                cs_off = s * 64
                x_s_off = s * n_count * 64
                cos_v = pto.vlds(cos_ub, cs_off, vf32)
                sin_v = pto.vlds(sin_ub, cs_off, vf32)
                cos_even, cos_odd = pto.vdintlv(cos_v, cos_v)
                sin_even, sin_odd = pto.vdintlv(sin_v, sin_v)
                for n in range(0, n_count, 1):
                    x_off = x_s_off + n * 64
                    x_v = pto.vlds(x_ub, x_off, vf32)
                    x_even, x_odd = pto.vdintlv(x_v, x_v)
                    y_even = pto.vsub(pto.vmul(x_even, cos_even, mask_pair), pto.vmul(x_odd, sin_even, mask_pair), mask_pair)
                    y_odd = pto.vadd(pto.vmul(x_odd, cos_odd, mask_pair), pto.vmul(x_even, sin_odd, mask_pair), mask_pair)
                    y_pack, _ = pto.vintlv(y_even, y_odd)
                    pto.vsts(y_pack, y_ub, x_off, mask_full)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(y_ub, y_gm, xy_dma_bytes, nburst=(1, xy_dma_bytes, xy_dma_bytes))
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(name="rope_mi_f16", target="a5", kernel_kind="vector", mode="explicit", insert_sync=False)
def rope_mi_f16(
    x_ptr: pto.ptr(pto.ui16, "gm"),
    cos_ptr: pto.ptr(pto.ui16, "gm"),
    sin_ptr: pto.ptr(pto.ui16, "gm"),
    y_ptr: pto.ptr(pto.ui16, "gm"),
    s_count: pto.i32,
    n_count: pto.i32,
    mode: pto.i32,
):
    gm_f16 = pto.ptr(pto.f16, "gm")
    ub_f16 = pto.ptr(pto.f16, "ub")
    x_gm = pto.castptr(x_ptr, gm_f16)
    cos_gm = pto.castptr(cos_ptr, gm_f16)
    sin_gm = pto.castptr(sin_ptr, gm_f16)
    y_gm = pto.castptr(y_ptr, gm_f16)

    cos_dma_bytes = _dma_round_up_32(s_count * 64 * 2)
    xy_dma_bytes = _dma_round_up_32(s_count * n_count * 64 * 2)

    cos_ub = pto.castptr(pto.const(0, dtype=pto.i64), ub_f16)
    sin_ub = pto.castptr(cos_dma_bytes, ub_f16)
    x_ub = pto.castptr(cos_dma_bytes + cos_dma_bytes, ub_f16)
    y_ub = pto.castptr(cos_dma_bytes + cos_dma_bytes + xy_dma_bytes, ub_f16)

    pto.mte_load(cos_gm, cos_ub, 0, cos_dma_bytes, nburst=(1, cos_dma_bytes, cos_dma_bytes))
    pto.mte_load(sin_gm, sin_ub, 0, cos_dma_bytes, nburst=(1, cos_dma_bytes, cos_dma_bytes))
    pto.mte_load(x_gm, x_ub, 0, xy_dma_bytes, nburst=(1, xy_dma_bytes, xy_dma_bytes))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    is_half_mode = mode == 0
    vf16 = pto.vreg_type(128, pto.f16)

    with pto.simd():
        if is_half_mode:
            mask = pto.pge_b16("PAT_VL32")
            for s in range(0, s_count, 1):
                cs_off = s * 64
                x_s_off = s * n_count * 64
                cos_lo = pto.vlds(cos_ub, cs_off, vf16)
                cos_hi = pto.vlds(cos_ub, cs_off + 32, vf16)
                sin_lo = pto.vlds(sin_ub, cs_off, vf16)
                sin_hi = pto.vlds(sin_ub, cs_off + 32, vf16)
                for n in range(0, n_count, 1):
                    x_off = x_s_off + n * 64
                    x_lo = pto.vlds(x_ub, x_off, vf16)
                    x_hi = pto.vlds(x_ub, x_off + 32, vf16)
                    y_lo = pto.vsub(pto.vmul(cos_lo, x_lo, mask), pto.vmul(sin_lo, x_hi, mask), mask)
                    y_hi = pto.vadd(pto.vmul(cos_hi, x_hi, mask), pto.vmul(sin_hi, x_lo, mask), mask)
                    pto.vsts(y_lo, y_ub, x_off, mask)
                    pto.vsts(y_hi, y_ub, x_off + 32, mask)
        else:
            mask_pair = pto.pge_b16("PAT_VL32")
            mask_full = pto.pge_b16("PAT_VL64")
            for s in range(0, s_count, 1):
                cs_off = s * 64
                x_s_off = s * n_count * 64
                cos_v = pto.vlds(cos_ub, cs_off, vf16)
                sin_v = pto.vlds(sin_ub, cs_off, vf16)
                cos_even, cos_odd = pto.vdintlv(cos_v, cos_v)
                sin_even, sin_odd = pto.vdintlv(sin_v, sin_v)
                for n in range(0, n_count, 1):
                    x_off = x_s_off + n * 64
                    x_v = pto.vlds(x_ub, x_off, vf16)
                    x_even, x_odd = pto.vdintlv(x_v, x_v)
                    y_even = pto.vsub(pto.vmul(x_even, cos_even, mask_pair), pto.vmul(x_odd, sin_even, mask_pair), mask_pair)
                    y_odd = pto.vadd(pto.vmul(x_odd, cos_odd, mask_pair), pto.vmul(x_even, sin_odd, mask_pair), mask_pair)
                    y_pack, _ = pto.vintlv(y_even, y_odd)
                    pto.vsts(y_pack, y_ub, x_off, mask_full)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(y_ub, y_gm, xy_dma_bytes, nburst=(1, xy_dma_bytes, xy_dma_bytes))
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(name="rope_mi_bf16", target="a5", kernel_kind="vector", mode="explicit", insert_sync=False)
def rope_mi_bf16(
    x_ptr: pto.ptr(pto.ui16, "gm"),
    cos_ptr: pto.ptr(pto.ui16, "gm"),
    sin_ptr: pto.ptr(pto.ui16, "gm"),
    y_ptr: pto.ptr(pto.ui16, "gm"),
    s_count: pto.i32,
    n_count: pto.i32,
    mode: pto.i32,
):
    gm_f16 = pto.ptr(pto.f16, "gm")
    ub_f16 = pto.ptr(pto.f16, "ub")
    gm_bf16 = pto.ptr(pto.bf16, "gm")
    ub_bf16 = pto.ptr(pto.bf16, "ub")
    x_gm = pto.castptr(x_ptr, gm_bf16)
    cos_gm = pto.castptr(cos_ptr, gm_f16)
    sin_gm = pto.castptr(sin_ptr, gm_f16)
    y_gm = pto.castptr(y_ptr, gm_bf16)

    cos_dma_bytes = _dma_round_up_32(s_count * 64 * 2)
    xy_dma_bytes = _dma_round_up_32(s_count * n_count * 64 * 2)

    cos_ub = pto.castptr(pto.const(0, dtype=pto.i64), ub_f16)
    sin_ub = pto.castptr(cos_dma_bytes, ub_f16)
    x_ub = pto.castptr(cos_dma_bytes + cos_dma_bytes, ub_bf16)
    y_ub = pto.castptr(cos_dma_bytes + cos_dma_bytes + xy_dma_bytes, ub_bf16)

    pto.mte_load(cos_gm, cos_ub, 0, cos_dma_bytes, nburst=(1, cos_dma_bytes, cos_dma_bytes))
    pto.mte_load(sin_gm, sin_ub, 0, cos_dma_bytes, nburst=(1, cos_dma_bytes, cos_dma_bytes))
    pto.mte_load(x_gm, x_ub, 0, xy_dma_bytes, nburst=(1, xy_dma_bytes, xy_dma_bytes))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    is_half_mode = mode == 0
    vf32 = pto.vreg_type(64, pto.f32)
    vf16_128 = pto.vreg_type(128, pto.f16)
    vbf16_128 = pto.vreg_type(128, pto.bf16)

    with pto.simd():
        if is_half_mode:
            mask16_all = pto.pset_b16("PAT_ALL")
            mask32_half = pto.pge_b32("PAT_VL32")
            for s in range(0, s_count, 1):
                cs_off = s * 64
                x_s_off = s * n_count * 64
                cos_lo_16 = pto.vlds(cos_ub, cs_off, vf16_128, dist="UNPK_B16")
                sin_lo_16 = pto.vlds(sin_ub, cs_off, vf16_128, dist="UNPK_B16")
                cos_hi_16 = pto.vlds(cos_ub, cs_off + 32, vf16_128, dist="UNPK_B16")
                sin_hi_16 = pto.vlds(sin_ub, cs_off + 32, vf16_128, dist="UNPK_B16")
                cos_lo = pto.vcvt(cos_lo_16, pto.f32, mask16_all, part="EVEN")
                sin_lo = pto.vcvt(sin_lo_16, pto.f32, mask16_all, part="EVEN")
                cos_hi = pto.vcvt(cos_hi_16, pto.f32, mask16_all, part="EVEN")
                sin_hi = pto.vcvt(sin_hi_16, pto.f32, mask16_all, part="EVEN")
                for n in range(0, n_count, 1):
                    x_off = x_s_off + n * 64
                    x_lo_16 = pto.vlds(x_ub, x_off, vbf16_128, dist="UNPK_B16")
                    x_hi_16 = pto.vlds(x_ub, x_off + 32, vbf16_128, dist="UNPK_B16")
                    x_lo = pto.vcvt(x_lo_16, pto.f32, mask16_all, part="EVEN")
                    x_hi = pto.vcvt(x_hi_16, pto.f32, mask16_all, part="EVEN")

                    y_lo_f32 = pto.vsub(pto.vmul(cos_lo, x_lo, mask32_half), pto.vmul(sin_lo, x_hi, mask32_half), mask32_half)
                    y_hi_f32 = pto.vadd(pto.vmul(cos_hi, x_hi, mask32_half), pto.vmul(sin_hi, x_lo, mask32_half), mask32_half)

                    y_lo_16 = pto.vcvt(y_lo_f32, pto.bf16, mask32_half, part="EVEN", rnd="R", sat="SAT")
                    y_hi_16 = pto.vcvt(y_hi_f32, pto.bf16, mask32_half, part="EVEN", rnd="R", sat="SAT")
                    pto.vsts(y_lo_16, y_ub, x_off, mask32_half, dist="PK_B32")
                    pto.vsts(y_hi_16, y_ub, x_off + 32, mask32_half, dist="PK_B32")
        else:
            mask16_all = pto.pset_b16("PAT_ALL")
            mask32_all = pto.pset_b32("PAT_ALL")
            for s in range(0, s_count, 1):
                cs_off = s * 64
                x_s_off = s * n_count * 64
                cos16 = pto.vlds(cos_ub, cs_off, vf16_128, dist="UNPK_B16")
                sin16 = pto.vlds(sin_ub, cs_off, vf16_128, dist="UNPK_B16")
                cos = pto.vcvt(cos16, pto.f32, mask16_all, part="EVEN")
                sin = pto.vcvt(sin16, pto.f32, mask16_all, part="EVEN")
                cos_even, cos_odd = pto.vdintlv(cos, cos)
                sin_even, sin_odd = pto.vdintlv(sin, sin)
                for n in range(0, n_count, 1):
                    x_off = x_s_off + n * 64
                    x16 = pto.vlds(x_ub, x_off, vbf16_128, dist="UNPK_B16")
                    x = pto.vcvt(x16, pto.f32, mask16_all, part="EVEN")
                    x_even, x_odd = pto.vdintlv(x, x)

                    y_even = pto.vsub(pto.vmul(x_even, cos_even, mask32_all), pto.vmul(x_odd, sin_even, mask32_all), mask32_all)
                    y_odd = pto.vadd(pto.vmul(x_odd, cos_odd, mask32_all), pto.vmul(x_even, sin_odd, mask32_all), mask32_all)

                    y_pack, _ = pto.vintlv(y_even, y_odd)
                    y_pack_16 = pto.vcvt(y_pack, pto.bf16, mask32_all, part="EVEN", rnd="R", sat="SAT")
                    pto.vsts(y_pack_16, y_ub, x_off, mask32_all, dist="PK_B32")

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(y_ub, y_gm, xy_dma_bytes, nburst=(1, xy_dma_bytes, xy_dma_bytes))
    pto.pipe_barrier(pto.Pipe.ALL)


_COMPILED: dict[str, object] = {}


def is_supported_variant(mode: str, dtype: str, cycle: bool = False) -> bool:
    del cycle
    return dtype in {"f16", "bf16", "f32"} and mode in {"half", "interleave"}


def describe() -> str:
    return "mi"


def _kernel_for_dtype(dtype: str):
    if dtype == "f16":
        return "f16", rope_mi_f16
    if dtype == "bf16":
        return "bf16", rope_mi_bf16
    if dtype == "f32":
        return "f32", rope_mi_f32
    raise ValueError(f"unsupported mi dtype: {dtype}")


def prepare(dtype: str = "f16", force_rebuild: bool = False):
    del force_rebuild
    key, kernel = _kernel_for_dtype(dtype)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = kernel.compile()
        _COMPILED[key] = compiled
    return compiled


def launch(ref: dict, cycle: bool = False):
    import torch

    from common.torch_runtime import device_str, empty_npu, stream_ptr, sync

    dtype_name = ref["dtype"]
    mode_name = ref["mode"]
    mode = 0 if mode_name == "half" else 1

    if dtype_name == "f16":
        x = torch.from_numpy(ref["x"]).to(torch.float16).to(device_str())
        cos = torch.from_numpy(ref["cos"]).to(torch.float16).to(device_str())
        sin = torch.from_numpy(ref["sin"]).to(torch.float16).to(device_str())
        y = empty_npu(ref["y"].shape, torch.float16)
        compiled = prepare("f16")
    elif dtype_name == "f32":
        x = torch.from_numpy(ref["x"]).to(torch.float32).to(device_str())
        cos = torch.from_numpy(ref["cos"]).to(torch.float32).to(device_str())
        sin = torch.from_numpy(ref["sin"]).to(torch.float32).to(device_str())
        y = empty_npu(ref["y"].shape, torch.float32)
        compiled = prepare("f32")
    elif dtype_name == "bf16":
        x = torch.from_numpy(ref["x"]).to(torch.bfloat16).to(device_str())
        cos = torch.from_numpy(ref["cos"]).to(torch.float16).to(device_str())
        sin = torch.from_numpy(ref["sin"]).to(torch.float16).to(device_str())
        y = empty_npu(ref["y"].shape, torch.bfloat16)
        compiled = prepare("bf16")
    else:
        raise ValueError(f"unsupported mi launch dtype: {dtype_name}")

    s_count, n_count = [int(v) for v in ref["params"]]

    compiled[1, stream_ptr()](
        x.data_ptr(),
        cos.data_ptr(),
        sin.data_ptr(),
        y.data_ptr(),
        s_count,
        n_count,
        mode,
    )
    sync()
    return y


def cache_tag() -> str:
    return describe()
