# 041 - Matrix Vector Multiply

## Overview

This example multiplies a dense matrix by a dense vector. It includes a simple one-thread-per-row baseline and a variant that stages vector tiles in shared memory to reuse them across the block.

## What this example teaches

- how matrix-vector multiply maps naturally to one output row per thread
- why even a modest amount of caching can help when every row reuses the same vector
- how to validate a dense linear-algebra kernel against a CPU reference

## CUDA concepts involved

- 1D block mapping over rows
- shared-memory caching
- repeated vector reuse
- dense floating-point validation

## Kernel mapping

- each thread computes one output row
- the naive kernel reads the entire vector directly from global memory
- the cached kernel loads one vector tile per block into shared memory, then every row in the block reuses it
- launch shape: `blocks = ceil(rows / block_size)`, `threads = block_size`

## Memory behavior

- matrix reads are row-major within a thread, but neighboring threads walk different rows
- the vector is reused for every row, so staging vector tiles in shared memory reduces repeated global loads
- synchronization is required before and after each shared-memory vector tile is consumed

## Correctness approach

- deterministic matrix and vector values come from fixed seeds
- a CPU reference computes the dense matrix-vector product
- both GPU kernels must match the CPU result within `1e-5`

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 1024 --block-size 256
.\example.exe --bench --size 4096 --warmup 5 --iters 20 --block-size 256
```

## Expected output

- `Validation: PASS`
- benchmark mode reports matrix size, timing, row throughput, and effective bandwidth

## Common mistakes

- returning early from a thread block before all threads reach the shared-memory synchronization points
- assuming matrix-vector multiply has the same reuse behavior as matrix-matrix multiply
- forgetting that the vector tile size is tied to the block size in this teaching kernel

## Possible optimizations / next step

- compare larger row counts and column counts to see when vector caching matters
- try warp-level cooperation for one row
- continue to `042_naive-matrix-multiply` and `043_tiled-matrix-multiply`
