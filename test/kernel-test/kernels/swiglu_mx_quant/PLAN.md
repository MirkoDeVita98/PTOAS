# swiglu_mx_quant — 1:1 VMI port + cycle comparison (WIP)

Goal: port the CCE `swiglu_mx_quant` kernel to VMI (PTODSL) 1:1 and compare cannsim
cycles vs CCE, using the kernel-test framework (like `rope`). Branch: `swiglu-mx-quant-vmi`
(in a5-kernel-standalone); kernel-test scaffold here in ptoas_feature_vmi.

## CCE source (reference)
a5-kernel-standalone/cce/swiglu_mx_quant/csrc/swiglu_mx_quant_kernel.cpp (1064 lines)
Copied into `cce/` here (kernel + smx_cce_shim.h + smx_tiling_data.h).

### Compute functions (all `__VEC_SCOPE__`, bf16 in / f32 compute):
1. ComputeVfSwigluV1 (L58)  — sigmoid gate: neg=-x1; e=exp(neg); s=x1/(1+e) [vdiv]; out=s*x2. mode 0.
2. ComputeVfSwigluV2 (L165) — clamped/alpha variant (vmaxs clamp, negAlpha, bias). swigluMode 1.
3. ComputeVfMaxExpVf (L259) — per-32-block max |value| exponent (for MX scale).
4. ComputeVfMaxExpVfBLAS (L320) — scale_alg=1 variant (cublas-style).
5. ComputeScale (L354) — scale_alg=0 (OCP): derive e8m0 scale from maxExp.
6. ComputeScaleBLAS (L429) — scale_alg=1 (BROKEN upstream; skip / OCP only).
7. ComputeDataFP4 (L516) — quantize f32 -> fp4 (e2m1/e1m2) with scale, bit-pack.
8. ComputeDataF8  (L627) — quantize f32 -> fp8 (e4m3/e5m2) with scale.
9. swiglu_mx_quant_kernel (L737) — entry: multicore tiling, double-buffer, DMA, orchestration.
   template<bool IS_BF16,int OUT_KIND,int ROUND_MODE,int SCALE_ALG>.

### ABI (extern C)
call_swiglu_mx_quant_<dtype>_<outkind>_<round>_<scalealg>(
    void* stream, uint8_t* x, uint8_t* group_index, uint8_t* y,
    uint8_t* mxscale, uint8_t* tiling_data, uint32_t blockDim)
Recommended first target: bf16_e4m3_rint_ocp (fp8, scale_alg=0=OCP, the working/verified path).

### Key constants (smx_cce_shim.h)
VL_B32=64, QUANT_ONCE_NUM=256, SCALE_ONCE_NUM=8, X_ONCE_NUM=512, UB_SIZE=262144,
OUT_E4M3=0/E5M2=1/E2M1=2/E1M2=3; TPL_RINT=1/ROUND=0/FLOOR=4; scale_alg 0=OCP,1=cublas.

## CCE -> VMI op mapping (all available in pto.vmi)
vlds(...,UNPK_B16)      -> pto.vmi.vload (size=64) [+ vcvt to f32]
vcvt(f32<-bf16)         -> pto.vmi.vcvt / extf ; vcvt(bf16<-f32) -> vcvt/truncf
vmuls / vmul            -> pto.vmi.vmuls / vmul
vadds                   -> pto.vmi.vadds
vexp                    -> pto.vmi.vexp
vdiv                    -> pto.vmi.vdiv
vmax / vmaxs            -> pto.vmi.vmax / vmaxs
vcmax (block max)       -> pto.vmi.vcmax
vbr (broadcast const)   -> pto.vmi.vbrc / constant
exponent bit-extract (shr/and on int) for MX scale -> pto.vmi integer ops
  (andi/ori/xori, extsi/extui, shift ops); VERIFY exact shift op names in pto.vmi.
vsts(...,PK_B32)        -> pto.vmi.vstore
pset_b32/plt_b32/plt_b16 masks -> pto.vmi.create_mask (size=64), half masks as needed

## Harness (kernel-test framework — mirror kernels/rope/)
Files to add under kernels/swiglu_mx_quant/:
- __init__.py, backends.py (register cce+vmi)
- tile_config.py  (dtypes: bf16/f16; OUT_KINDs; tolerances; tile: dim0 x dim1)
- spec.py         (list_cases per {dtype,out_kind,round,scale_alg}; verify_case)
- reference.py    (numpy/torch golden: swiglu then MX-quant to fp8/fp4 + e8m0 scale;
                   REUSE logic from cce/swiglu_mx_quant/test/test_equivalence.py +
                   common/tiling.py for tiling params)
- runtime.py      (LaunchArgs: x, y(uint8), scale(uint8), tiling bytes; prepare_launch_args)
- cce/CMakeLists.txt (adapt kernels/rope/cce/CMakeLists.txt: project swiglu_mx_quant_cce,
                      add_library from swiglu_mx_quant_kernel.cpp; same bisheng flags)
- cce/backend.py  (build .so via cmake; ctypes bind call_swiglu_mx_quant_*; launch with
                   stream,x,group_index,y,mxscale,tiling,blockDim)
- vmi/backend.py  (the PTODSL 1:1 port; source-backed .vmi.pto or traced)

Tiling params come from cce/swiglu_mx_quant/common/tiling.py (compute SwigluMxQuantTilingData).

## Run (cannsim transport, cycles + correctness)
cd test/kernel-test
scripts/run_sim.sh --output sim_outputs/smx-cce run.py -- --op swiglu_mx_quant --workflow cycle --backend cce --case bf16_e4m3_rint_ocp
scripts/run_sim.sh --output sim_outputs/smx-vmi run.py -- --op swiglu_mx_quant --workflow cycle --backend vmi --case bf16_e4m3_rint_ocp
# correctness: --workflow correctness  (RVEC span in <out>/cannsim_*/report/trace_core0.json)
Env: PTOAS_BIN=$(which ptoas) (v0.1.1 wheel), ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0.

## Build order (incremental, test each step)
1. CMakeLists + cce/backend.py -> build librope-style .so; smoke-load symbols.
2. tiling.py + runtime.py + reference.py (fp8 e4m3 ocp golden) + spec.py + register op.
3. CCE baseline: run --backend cce --workflow correctness then cycle -> record CCE RVEC.
4. VMI port: start with ComputeVfSwigluV1 (sigmoid gate) as .vmi.pto; then MaxExp, Scale,
   ComputeDataF8; wire the entry (tiling/DMA). Verify correctness after each, then cycle.
5. Compare VMI vs CCE RVEC; document (like rope: even/odd vs pack/unpack learnings).

## Status
- [x] Branch swiglu-mx-quant-vmi created
- [x] CCE kernel + shim + tiling copied into cce/
- [x] Full scope + op mapping captured (this file)
- [ ] CMakeLists + cce backend  (next)
- [ ] tiling/runtime/reference/spec + op registration
- [ ] CCE cannsim baseline
- [ ] VMI port (8 fns + entry)
- [ ] correctness + cycle comparison

## Per-tile compute analysis (primary port target: bf16, OUT_E4M3, OCP, swigluMode 0)
The entry is a HW kernel (DMA, get_buf/rls_buf, event flags, double-buffer, multicore).
Like rope, the meaningful cycle comparison is the per-tile VECTOR compute (RVEC span),
which the CCE report says dominates (compute-bound on vexp/vdiv). Port that pipeline:

### (1) ComputeVfSwigluV1  [64-lane f32, bf16 in]  -- the sigmoid gate
  vlds x1,x2 (bf16,size64,UNPK) -> vcvt f32
  neg = x1 * -1 ; e = exp(neg) ; d = e + 1 ; sig = x1 / d ; out = sig * x2
  vcvt f32->bf16 ; store swiglu. VMI: vload/vcvt/vmuls/vexp/vadds/vdiv/vmul/vcvt/vstore. ALL trivially available.

### (2) ComputeVfMaxExpVf  [256 elems -> 8 per-32-block max exponents]
  vlds v0,v1 (DINTLV_B16, 256) ; vand v0,v1 with BF16_EXP_MASK(0x7F80) ; vmax(v0,v1)
  vcgmax  (grouped max, reduce each 32-block -> block max exponent) ; vstus 8 scale slots.
  VMI RISK: vcgmax (grouped/segmented max reduce over 32-lane blocks) + vstus/vstas
  (append-store to align buffer). Confirm vmi has vcgmax/segmented-reduce + strided store.

### (3) ComputeScale (OCP)  [maxExp -> mxScale byte + halfScale bf16 multiplier]
  sharedExp = maxExp - fEmax ; scaleValue = sharedExp >> 7 (e8m0 output scale)
  halfScale = BF16_EXP_BIAS - sharedExp  (bf16-exp form of 2^-k reciprocal scale)
  inf/zero/special vsel guards. store mxScale (PK_B16) + halfScale (NORM_B16).
  VMI: vsub/vshrs(shift)/vsel/vcmp_ne/le/eq + vbrc consts. Confirm vmi integer shift+select+cmp.

### (4) ComputeDataF8 (e4m3)  [256 bf16 -> 256 fp8 bytes, per-block scaled]
  vlds_e2b_b16 halfScale (broadcast 1 scale per 32-block!) ; vlds_x2 x0,x1
  x0*=halfScale ; x1*=halfScale (bf16) ; vcvt bf16->fp32 even/odd (4 vecs)
  vcvt fp32->fp8e4m3 into PART_P0/P1/P2/P3 subregs ; MergeAndStoreFp8NormB8.
  VMI RISK: (a) e2b block-broadcast load of scale; (b) fp32->fp8 with P0..P3 lane
  interleave + NORM_B8 merge store. Confirm vmi fp8 cvt granularity + pack/merge story.
  THIS is where a rope-like pack/unpack cycle penalty may appear -> the finding to measure.

### ComputeDataFP4 (secondary, OUT_E2M1): adds vintlv/vpack fp4 x2 packing -> more pack ops.

## Refined scope / order
1. CCE harness + baseline cycles for bf16_e4m3_rint_ocp (single representative tile).  <- do first
2. VMI port pipeline (1)->(4); verify correctness each stage; then RVEC cycle vs CCE.
3. Then fp4 (OUT_E2M1) + f16 input variants + swigluMode 1 (V2) to complete "full port".

## Non-elementwise ops to confirm exist in pto.vmi (blockers if missing)
  vcgmax / segmented-max-reduce over 32-lane blocks   (MaxExp)
  block-broadcast load (vlds_e2b_b16 equiv)           (DataF8 scale)
  fp32->fp8 e4m3/e5m2 cvt + P0..P3 merge/NORM_B8 store (DataF8)
  fp4 e2m1/e1m2 cvt + x2 nibble pack (PK4_B32)         (DataFP4)
  vstus/vstas append-to-align store                    (MaxExp)
  integer vand/vshrs/vsub/vsel/vcmp on u16/u32         (Scale)

## KEY PRECEDENT (de-risks the whole port): vmi-demo per_block_cast_vmi.py
/home/mdevita/vmi-demo/quant/per_block_cast/vmi/per_block_cast_vmi.py is the EXACT
VMI Python-DSL idiom for MX-fp8 quant. Copy its structure. VMI is written in Python DSL
(@pto.jit), NOT raw .vmi.pto MLIR. Key ops used (all confirmed working there):
  pto.jit(name=,target="a5",backend="vpto",mode="explicit",kernel_kind="vector",insert_sync=False)
  pto.ptr(dtype,"gm"); pto.castptr(pto.const(UB_OFF,ui64), pto.ptr(dtype,"ub"))
  pto.mte_gm_ub / mte_ub_gm (DMA); pto.set_flag/wait_flag(Pipe.MTE2,Pipe.V,event_id=)
  pto.vmi.create_mask(n,size=); vbrc; vload(ptr,off,size=[,dist_mode="brc"]); vcvt(v,f32)
  vabs; vmax; vcmax (full reduce); vand; vshrs; vshls; vsub; vsel; vcmps(v,imm,mask,"eq/lt")
  vinterpret_cast(v,ui32); vmul; vstore; vcvt(x_f32, pto.f8e4m3, rounding="R", saturate="SAT")
  pto.mem_bar(pto.BarrierType.VST_VLD)
e8m0/recip bit math (fp32): exp=(bits&0x7F800000)>>23; shared=exp-emax; scale=e8<<23;
  recip=(254-e8)<<23; guards for zero/nan/too_small via vsel.

DIFFERENCES swiglu_mx_quant vs per_block_cast:
  - MX block = 32 elements (QUANT_ONCE_NUM=256 / SCALE_ONCE_NUM=8 => 32/scale), NOT 1024.
    Need per-32-block max -> use vcgmax (segmented) instead of single vcmax. (CCE: vcgmax)
  - SwiGLU sigmoid-gate PREFIX before quant: out = (x1*sigmoid(x1))*x2  [swigluMode 0]
  - bf16/f16 exponent path (BF16_EXP_MASK 0x7F80, bias 127, SHR_NUM_FOR_BF16=7) vs fp32.
  - fp8 e4m3 emax=8 (YMaxExpForOutKind), fp4 e2m1 emax differs.

Harness files still mirror kernels/rope/ (spec/reference/runtime/tile_config/backends/
cce backend+CMakeLists/vmi backend), per user choice = ptoas kernel-test.

## FRAMEWORK CONTRACT (from run.py -> kernel_test.cli/registry)
Discovery: kernels/<op>/ needs __init__.py + backends.py; loader calls get_operator_spec().
Files: __init__.py (get_operator_spec via make_operator_spec name/default_backend/backend_names/
  create_backend/list_cases/verify/cycle_fields), backends.py (create_backend(name)->BackendAdapter),
  spec.py (list_cases(workflow)->{id:case dict}, verify_case(id,case,out)->CaseResult, cycle_fields),
  tile_config.py (constants, TileConfig dataclass, TOLERANCE, sim_fn_name(mode,dtype,cycle)->symbol),
  runtime.py (LaunchArgs dataclass, prepare_launch_args(case,cycle=)->args, artifact_case_dir),
  reference.py (generate_case/generate_all -> {id:case dict}, numpy golden),
  cce/backend.py (BackendAdapter: _build_lib via cmake -S cce -B build -DASCEND_HOME_PATH -DASCEND_DRIVER_PATH,
     cmake --build --target swiglu_mx_quant_cce ; ctypes.CDLL(libswiglu_mx_quant_cce.so);
     bind _SIM_SYMBOLS with argtypes; _launch calls fn(stream_ptr(), *bufptrs [, blockDim]); sync()),
  cce/CMakeLists.txt (mirror rope: project swiglu_mx_quant_cce, add_library from .cpp, bisheng CXX,
     include dirs, -xcce/-mllvm/--cce-aicore-arch=dav-c310-vec/--cce-simd-vf-fusion=false/-DREGISTER_BASE,
     --cce-fatobj-link -Wl,-soname),
  cce/swiglu_mx_quant_cce_kernel.cpp (already have swiglu_mx_quant_kernel.cpp + inc/ shims),
  vmi/backend.py (PTODSL @pto.jit like per_block_cast_vmi.py; JIT-compiled, no .vmi.pto needed at runtime;
     build_artifact_plan emits generated/vmi/<case>/vmi.pto via --emit-mlir),
  cycle_metrics.py (CYCLE_REPORTER=CycleReporterSpec, get_cycle_reporter, main=run_cycle_report).
Framework imports: kernel_test.registry.{OperatorSpec,make_operator_spec},
  kernel_test.backends.{BackendAdapter,RunPurpose,ArtifactPlan}, kernel_test.results.CaseResult,
  kernel_test.npu_runtime.{ensure_runtime,stream_ptr,sync,empty_npu,device_str},
  kernel_test.cycle_reporting.{CycleReporterSpec,run_cycle_report}.
ABI: call_swiglu_mx_quant_<suffix>(void* stream, u8* x, u8* group_index, u8* y, u8* mxscale,
  u8* tiling_data, u32 blockDim) -> argtypes=[c_void_p]*6+[c_uint32].
Run cycle:  scripts/run_sim.sh --output sim_outputs/swiglu_mx_quant/cce/<case> run.py -- \
              --op swiglu_mx_quant --backend cce --workflow cycle --case <case>
Cycle metric = RVEC span from cannsim_*/report/trace_core0.json (framework parses it).
Tiling struct filled by a5 cce/swiglu_mx_quant/common/tiling.py (SwigluMxQuantTilingData) -> port to numpy bytes.
Golden = silu-gate + per-32-block MX quant to fp8/fp4 (adapt vmi-demo .../per_block_cast/cce/ref/golden.py).
