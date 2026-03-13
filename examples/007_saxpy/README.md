# 007 - SAXPY

- Track: `Foundations`
- Difficulty: `Beginner`
- Status: `✅ fully mature`
- Maturity: `Level 5 - optimized / variant-ready`

## Goal

Compute `a * x + y` on the GPU using a clean CUDA kernel, then validate the result against a CPU reference.

## Why This Example Matters

SAXPY is a classic bridge between beginner kernels and linear algebra workloads. It is still simple to reason about, but it starts to feel more like a real numeric kernel than plain vector addition.

## CUDA Concepts Taught

- 1D thread mapping
- scalar kernel parameters
- CPU reference validation
- benchmark mode for a lightweight arithmetic kernel

## Prerequisites

- `002_vector-addition`

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 8192
```

```powershell
.\example.exe --bench --size 1048576 --warmup 5 --iters 20
```

## Expected Output

- Prints `PASS` when GPU output matches the CPU reference within tolerance.
- Benchmark mode prints timing and estimated throughput.

## Correctness Notes

- Inputs are deterministic from a fixed seed.
- Validation compares every output element against the CPU result.

## Benchmark Notes

- SAXPY is still largely memory-bound, but it carries slightly more arithmetic work than plain vector addition.

## Likely Bottlenecks

- global memory bandwidth
- launch overhead at small problem sizes

## Next Optimization Steps

- compare with a grid-stride-loop version
- measure performance with different block sizes
- explore fused kernels that combine multiple vector operations
