# FlashAttention compile and benchmark

This directory contains two PTO FlashAttention variants:

- `fa_140tflops.pto`
- `fa_patched_s1_256_q3072_s0_8192.pto`

## Requirements

- Run inside the configured Ascend/CANN container environment.
- `ptoas` and `bisheng` must already be available in `PATH`.
- `/sources/pto-isa/include` must exist.
- Python benchmark requires `torch_npu==2.9.0`.

## Compile

From this directory, run:

```bash
bash compile_flashattention.sh
```

This builds:

- `/tmp/fa_140tflops.so`
- `/tmp/compiler_team_fa.so`

## Benchmark

After compiling, run:

```bash
python3 benchmark_flashattention.py
```

The benchmark compares both PTO kernels against `torch_npu.npu_fused_infer_attention_score`, checks correctness against both fp32 reference attention and torch_npu output, and reports latency, TFLOP/s, and speedup.