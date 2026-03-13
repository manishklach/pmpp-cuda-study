# 043 - Tiled Matrix Multiply

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `✅ fully mature`
- Maturity: `Level 6 - polished teaching example`

## Goal

Multiply two dense square matrices using shared-memory tiling, then validate the result against a CPU reference.

## Why This Example Matters

This is one of the most important CUDA study examples in the repo. It brings together thread mapping, tiling, synchronization, and memory reuse in a form that is easy to compare against a naive baseline.

## CUDA Concepts Taught

- shared-memory tiles
- block-level cooperation
- `__syncthreads()`
- reduced global-memory traffic
- benchmark mode for an optimized teaching kernel

## Prerequisites

- `042_naive-matrix-multiply`

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 32
```

```powershell
.\example.exe --bench --size 128 --warmup 5 --iters 10
```

## Expected Output

- Prints `PASS` when GPU output matches the CPU result within tolerance.
- Benchmark mode prints timing and output-element throughput.

## Correctness Notes

- The CPU reference uses the same square-matrix shape as the GPU kernel.
- Validation uses an absolute tolerance of `1e-4`.

## Benchmark Notes

- This kernel should outperform the naive version for larger sizes because each tile load is reused by multiple threads.

## Likely Bottlenecks

- shared-memory capacity limits
- tile size choice
- synchronization overhead

## Next Optimization Steps

- compare against `042_naive-matrix-multiply`
- try tile sizes 8, 16, and 32
- add rectangular matrix support and block-shape experiments
