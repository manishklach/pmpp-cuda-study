# 043 - Tiled Matrix Multiply

## Overview

This example multiplies two dense square matrices using shared-memory tiling. It is the classic CUDA optimization step after the naive global-memory matrix multiply.

## What this example teaches

- how tiling improves data reuse
- how blocks cooperate through shared memory
- why matrix multiply is such a strong CUDA teaching example

## CUDA concepts involved

- 2D tiles
- shared memory
- cooperative loading
- synchronization with `__syncthreads()`
- reuse-driven optimization

## Kernel mapping

- each block computes one 16x16 output tile of `C`
- threads cooperatively load a tile of `A` and a tile of `B`
- each thread accumulates one output element using those shared tiles
- launch shape: `blocks = ceil(size / 16) x ceil(size / 16)`, `threads = 16 x 16`

## Memory behavior

- global loads are staged into shared-memory tiles
- each loaded value is reused by many multiply-adds before the next tile is fetched
- tiling improves reuse because the same `A` row tile and `B` column tile feed multiple output elements instead of being reread from global memory by every thread

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

- forgetting one of the two synchronization points around shared-memory tile use
- assuming tiling helps without understanding that the win comes from reuse, not from shared memory by itself
- choosing a tile size that does not fit well with shared-memory capacity or occupancy

## Possible optimizations / next step

- compare timing directly against `042_naive-matrix-multiply`
- try tile sizes 8, 16, and 32
- explore register tiling, vectorized loads, or warp-level matrix instructions after this baseline is clear
