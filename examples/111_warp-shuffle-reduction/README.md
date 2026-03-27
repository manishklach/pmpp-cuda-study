# 111 - Warp Shuffle Reduction

## Overview

Reduce values using warp shuffle intrinsics before a small shared-memory merge.

## What this example teaches

- how warp shuffles replace part of the shared-memory reduction tree
- why only warp leaders need to spill partial sums to shared memory
- how to keep a block reduction readable while using modern warp intrinsics

## CUDA concepts involved

- `__shfl_down_sync`
- warp-local reduction
- block-wide merge through shared memory

## Kernel mapping

- each thread accumulates a grid-stride subset of the input
- each warp reduces its local totals using shuffles
- warp leaders write one partial per warp, then the first warp finishes the block reduction

## Memory behavior

- global reads are regular and coalesced
- shuffles reduce the amount of shared-memory traffic inside each warp
- shared memory is still used to merge warp partials into one block result

## Correctness approach

- deterministic float input
- CPU reference uses scalar accumulation
- GPU result must match within `1e-3`

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 65536
.\example.exe --bench --size 1048576 --warmup 5 --iters 20
```

## Expected output

- `Validation: PASS`
- benchmark mode reports input size, timing, and throughput

## Common mistakes

- forgetting that shuffle operations only communicate inside one warp
- assuming a shuffle-only reduction can merge multiple warps without shared memory or another step
- ignoring the block-size assumptions behind the shared partial array

## Possible optimizations / next step

- compare directly with a shared-memory reduction baseline
- extend the idea to warp-local scans
- study how the warp merge changes with different block sizes
