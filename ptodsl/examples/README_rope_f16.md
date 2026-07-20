# RoPE f16 — VMI kernels + CCE reference (GPT-J / interleave)

Three functionally-identical RoPE f16 interleave kernels (all maxDiff=0.000000 vs the
numpy reference). They differ only in how the rotation is expressed and where it runs.

| File | Backend | Rotation form | RVEC cycles (tile s=15, n=32) |
|------|---------|---------------|-------------------------------|
| `rope_f16.vmi.pto`      | VMI (PTODSL) | even/odd: `y_even=x_even*cos-x_odd*sin`, `y_odd=x_odd*cos+x_even*sin` (4 mul, 1 sub, 1 add) | **4077** |
| `rope_f16_cce.vmi.pto`  | VMI (PTODSL) | rotate-partner, 1:1 with CCE: `rot(x)=[-x_odd,x_even]` via `vdintlv -> vneg -> vintlv`, then `y=x*cos+rot*sin` (2 mul, 1 add) | **4639** |
| `rope_cce_compute.h`    | CCE (hand-written MI) | rotate-partner (`ComputeF16`, negate via `*-1`) | **3036** |

All three use `size=64` contiguous loads (fast). The VMI kernels here are the `size=64`
`pto.vmi.vload` + native `vdintlv`/`vintlv` form.

## Why the 1:1-CCE VMI form is slower than the even/odd VMI form

`rope_f16_cce.vmi.pto` mirrors CCE `ComputeF16` exactly and does **half** the muls (2 vs 4),
yet is ~14% slower (4639 vs 4077). The `vneg` on the deinterleaved `x_odd` followed by
re-`vintlv` makes the VMI layout pipeline insert `RV_VPACK` + `RV_VZUNPACK` conversions
(480 each); the even/odd form avoids them by staying in the deinterleaved layout. CCE hits
3036 with the same math because hand-written MI controls the physical layout directly.
Until the VMI layout pipeline can carry a value through `vdintlv -> elementwise -> vintlv`
without pack/unpack, the even/odd form (`rope_f16.vmi.pto`) is the faster VMI choice.

## How to run (correctness + cycles, via cannsim)

The runnable backends live in `test/kernel-test/kernels/rope/{cce,vmi}`. From `test/kernel-test`:

```bash
# Correctness — expect: PASS, maxDiff=0.000000
scripts/run_sim.sh --output sim_outputs/rope-cce-corr run.py -- \
  --op rope --workflow correctness --backend cce --case f16_interleave
scripts/run_sim.sh --output sim_outputs/rope-vmi-corr run.py -- \
  --op rope --workflow correctness --backend vmi --case f16_interleave

# Cycles — RVEC span is in <output>/cannsim_*/report/trace_core0.json
scripts/run_sim.sh --output sim_outputs/rope-cce-cyc run.py -- \
  --op rope --workflow cycle --backend cce --case f16_interleave   # -> 3036
scripts/run_sim.sh --output sim_outputs/rope-vmi-cyc run.py -- \
  --op rope --workflow cycle --backend vmi --case f16_interleave   # -> 4077 (even/odd)
```

The CCE reference kernel run above is `rope_cce_compute.h` (included here for the 1:1
comparison; the runnable copy + launcher live under `test/kernel-test/kernels/rope/cce/`).

## Compile / inspect the VMI examples (no NPU required)

```bash
ptoas rope_f16.vmi.pto     --pto-backend=vpto --enable-vmi -o /tmp/rope_f16.o
ptoas rope_f16_cce.vmi.pto --pto-backend=vpto --enable-vmi -o /tmp/rope_f16_cce.o
```

Verified with ptoas vmi-v0.1.1.
