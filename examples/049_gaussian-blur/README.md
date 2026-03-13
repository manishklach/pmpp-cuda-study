# 049 - Gaussian Blur

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Apply a 3x3 Gaussian blur to a square grayscale image on the GPU and validate the result against a CPU reference.

## Why This Example Matters

This is a compact image-processing kernel that makes 2D grids, neighborhood access, and border handling concrete. It is a strong bridge between basic image kernels and more advanced stencil-style workloads.

## CUDA Concepts Taught

- 2D thread mapping
- stencil-style neighborhood access
- shared constant-like kernel staging
- border clamping

## Prerequisites

- `012_rgb-to-grayscale`
- `043_tiled-matrix-multiply`

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 64
```

```powershell
.\example.exe --bench --size 256 --warmup 5 --iters 20
```

## Expected Output

- Prints `PASS` when GPU output matches the CPU blur within tolerance.
- Benchmark mode prints runtime and pixel throughput.

## Correctness Notes

- The image is generated deterministically from a fixed seed.
- Boundary handling uses clamped coordinates on both CPU and GPU.

## Benchmark Notes

- This is a useful first benchmark for neighborhood-based image kernels.

## Likely Bottlenecks

- redundant neighborhood loads from global memory
- memory access locality near tile boundaries

## Next Optimization Steps

- move the blur weights to constant memory
- tile the input image with halos in shared memory
- compare direct blur with a separable implementation
