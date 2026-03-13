# 023 - Sum Reduction

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `✅ fully mature`
- Maturity: `Level 6 - polished teaching example`

## Goal

Reduce a large float array to one sum using a shared-memory block reduction on the GPU and validate the result against a CPU reference.

## Why This Example Matters

Reduction is one of the most important PMPP patterns. It teaches how many threads cooperate to produce one answer, and it sets up later work on scan, histogramming, norms, and softmax-like kernels.

## CUDA Concepts Taught

- shared-memory reduction
- block partials
- synchronization
- staged aggregation
- benchmark mode for collective kernels

## Prerequisites

- `002_vector-addition`
- `007_saxpy`

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 65536 --block-size 256
```

```powershell
.\example.exe --bench --size 1048576 --warmup 5 --iters 20 --block-size 256
```

## Expected Output

- Prints `PASS` when the GPU result matches the CPU sum within tolerance.
- Benchmark mode prints timing and element throughput.

## Correctness Notes

- Inputs are deterministic.
- The GPU computes block partials on device, then the host accumulates those partials for the final result.
- Validation uses a scalar CPU reference sum.

## Benchmark Notes

- This version times the GPU partial-reduction pass.
- The final host accumulation is intentionally kept simple so the reduction structure is easy to inspect.

## Likely Bottlenecks

- shared-memory synchronization
- underutilization at small sizes
- extra host work for the final aggregation step

## Next Optimization Steps

- add a second GPU reduction pass
- compare against a warp-shuffle reduction
- measure different block sizes and occupancy tradeoffs
