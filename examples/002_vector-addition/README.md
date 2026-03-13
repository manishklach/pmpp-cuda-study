# 002 - Vector Addition

- Track: `Foundations`
- Difficulty: `Beginner`
- Status: `✅ fully mature`
- Maturity: `Level 6 - polished teaching example`

## Goal

Build a trustworthy CUDA hello-world example that demonstrates the full host/device workflow:

- allocate GPU memory
- copy host inputs to the device
- launch a simple kernel
- copy results back
- validate against a CPU reference

## Why This Example Matters

This is the cleanest starting point for CUDA study. It keeps the algorithm simple so you can focus on kernel launches, memory movement, and correctness.

## CUDA Concepts Taught

- 1D thread indexing
- global memory access
- host-to-device and device-to-host copies
- CPU reference validation
- benchmark mode for a memory-bound kernel

## Prerequisites

- `001_hello-world-kernel`

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

Correctness mode:

```powershell
.\example.exe --check --size 4096
```

Benchmark mode:

```powershell
.\example.exe --bench --size 1048576 --warmup 5 --iters 20
```

Benchmark mode with validation first:

```powershell
.\example.exe --check --bench --size 1048576 --warmup 5 --iters 20
```

## Expected Output

- Prints `PASS` when the GPU output matches the CPU reference within tolerance.
- In benchmark mode, prints average, minimum, and maximum runtime along with effective bandwidth and elements per second.

## Correctness Notes

- Inputs are generated deterministically from a fixed seed.
- Validation is elementwise against a CPU reference.
- Floating-point comparisons use a `1e-5` absolute tolerance.

## Benchmark Notes

- This kernel is primarily memory-bandwidth bound.
- Effective GB/s is estimated from reading two input vectors and writing one output vector.

## Likely Bottlenecks

- global memory bandwidth
- launch overhead for very small arrays

## Next Optimization Steps

- rewrite the kernel as a grid-stride loop
- compare pageable vs pinned host memory for transfer-heavy runs
- add a fused arithmetic variant and compare throughput
