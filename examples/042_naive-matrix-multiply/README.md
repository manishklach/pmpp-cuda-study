# 042 - Naive Matrix Multiply

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable baseline`

## Goal

Multiply two dense square matrices with a straightforward CUDA kernel and verify the result against a CPU reference.

## Why This Example Matters

This is the baseline dense GEMM example. It matters because the optimized tiled version only makes sense once the naive mapping and its global-memory cost are clear.

## CUDA Concepts Taught

- 2D thread/block mapping
- one-thread-per-output-element mapping
- baseline dense numeric kernels
- benchmark mode for a compute-heavy workload

## Prerequisites

- `002_vector-addition`
- `023_sum-reduction`

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

- Prints `PASS` when GPU and CPU matrix products match within tolerance.
- Benchmark mode prints timing and output-element throughput.

## Correctness Notes

- The example uses square matrices driven by a single `--size` value.
- Validation compares every output element against the CPU product.

## Benchmark Notes

- This version is intentionally naive and rereads matrix values from global memory many times.

## Likely Bottlenecks

- repeated global-memory loads
- low arithmetic intensity relative to optimized tiled variants

## Next Optimization Steps

- compare directly with `043_tiled-matrix-multiply`
- experiment with different block shapes
- add rectangular matrix support
