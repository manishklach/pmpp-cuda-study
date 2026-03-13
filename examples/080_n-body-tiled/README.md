# 080 - N Body Tiled

- Track: `Simulation`
- Difficulty: `Advanced`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Compute particle accelerations from an all-pairs N-body interaction using shared-memory tiling and validate the GPU result against a CPU reference.

## Why This Example Matters

This is a strong simulation example because it turns shared-memory reuse into a real workload. It also gives the repo a more mature flagship outside the usual vector and matrix kernels.

## CUDA Concepts Taught

- shared-memory tiling
- all-pairs interaction kernels
- reuse of loaded particle positions
- tolerance-based validation for floating-point simulation output

## Prerequisites

- `023_sum-reduction`
- `043_tiled-matrix-multiply`

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 128
```

```powershell
.\example.exe --bench --size 1024 --warmup 3 --iters 10
```

## Expected Output

- Prints `PASS` when the GPU accelerations match the CPU reference within tolerance.
- Benchmark mode prints runtime and particle throughput.

## Correctness Notes

- The example uses deterministic particle positions generated from a fixed seed.
- Validation compares all three acceleration components per particle.

## Benchmark Notes

- This example is compute-heavy compared with the beginner vector kernels.
- Runtime grows quickly with particle count because the workload is still all-pairs.

## Likely Bottlenecks

- arithmetic intensity of all-pairs force accumulation
- synchronization per shared-memory tile
- quadratic growth with particle count

## Next Optimization Steps

- compare against `079_n-body-naive`
- experiment with tile sizes and occupancy
- add force integration to build a multi-step simulation
