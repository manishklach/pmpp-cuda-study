# 112 - Warp Shuffle Scan

## Overview

Build a block-wide inclusive scan from warp-local scans plus scanned warp totals.

## What this example teaches

- how to use shuffle intrinsics for prefix propagation inside a warp
- how block-wide scans can be assembled from warp-level building blocks
- why one-block study kernels are a useful place to learn shuffle logic

## CUDA concepts involved

- `__shfl_up_sync`
- warp-local inclusive scan
- shared-memory warp-total carry propagation

## Kernel mapping

- one thread owns one input element
- each warp scans its local values with shuffles
- the first warp scans the warp totals, then each warp adds its carry

## Memory behavior

- input and output live in global memory
- warp-local communication stays in registers through shuffle operations
- shared memory only stores the per-warp totals

## Correctness approach

- deterministic integer input
- CPU reference computes the inclusive scan sequentially
- GPU output must match exactly

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 128
.\example.exe --bench --size 256 --warmup 5 --iters 20
```

## Expected output

- `Validation: PASS`
- benchmark mode reports scanned element count and timing

## Common mistakes

- forgetting that shuffles do not communicate between warps
- launching a non-warp-aligned block without handling inactive lanes carefully
- mixing inclusive and exclusive carry logic

## Possible optimizations / next step

- compare against a shared-memory scan baseline
- extend the design beyond one block
- reuse the block-wide scan in compaction or radix partitioning
