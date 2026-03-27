# 042 - Naive Matrix Multiply

## Overview

This example multiplies two dense square matrices using the straightforward global-memory formulation. It is the baseline that makes the tiled shared-memory version worth studying.

## What this example teaches

- how to map one thread to one output matrix element
- why naive matrix multiply rereads the same values many times
- how to benchmark a dense compute kernel after validating correctness

## CUDA concepts involved

- 2D grids and blocks
- output-space mapping
- global-memory-heavy dense compute
- CPU reference validation

## Kernel mapping

- each thread computes one `C[row, col]`
- blocks cover 16x16 output tiles
- every thread walks across one row of `A` and one column of `B`
- launch shape: `blocks = ceil(size / 16) x ceil(size / 16)`, `threads = 16 x 16`

## Memory behavior

- every multiply-add reads directly from global memory
- rows of `A` are reused across nearby columns, but this naive kernel does not cache them
- columns of `B` are reread by many threads, which is one reason the tiled version can help so much

## Correctness approach

- deterministic input matrices come from fixed seeds
- a CPU reference computes the same dense product
- the GPU output must match the CPU result within `1e-4`

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 64
.\example.exe --bench --size 256 --warmup 5 --iters 10
```

## Expected output

- `Validation: PASS`
- benchmark mode reports matrix dimension, timing, output-element throughput, and effective bandwidth

## Common mistakes

- mixing up row-major indexing for `A`, `B`, and `C`
- treating the naive kernel as a performance target instead of a correctness baseline
- forgetting that square matrix size grows total work cubically

## Possible optimizations / next step

- compare directly with `043_tiled-matrix-multiply`
- experiment with different block shapes
- extend the example to rectangular matrices
