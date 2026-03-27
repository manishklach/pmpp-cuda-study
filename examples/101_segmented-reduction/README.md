# 101 - Segmented Reduction

## Overview

Reduce values independently inside contiguous segments.

## What this example teaches

- how segmented reduction differs from a single global reduction
- why assigning one block per segment is a useful baseline
- how to validate grouped results against a CPU reference

## CUDA concepts involved

- block-level shared-memory reduction
- contiguous segment ownership
- synchronization inside a reduction tree

## Kernel mapping

- one block owns one segment
- each thread accumulates a strided subset of that segment
- thread 0 writes one reduced value per segment

## Memory behavior

- the segment values are read contiguously
- partial sums live in shared memory during the block reduction
- this version favors clarity over load balancing for highly irregular segments

## Correctness approach

- deterministic float input
- contiguous segment offsets generated on the host
- CPU reference reduces each segment independently

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 4096
.\example.exe --bench --size 8192 --warmup 5 --iters 20
```

## Expected output

- `Validation: PASS`
- benchmark mode reports input size, timing, and throughput

## Common mistakes

- mixing segment indices with global indices
- forgetting that all threads in a block must reach the same synchronization points
- assuming this one-block-per-segment baseline handles very long or badly imbalanced segments well

## Possible optimizations / next step

- support variable-length segments with better load balancing
- compare against warp-level segmented reductions
- connect this example to segmented scan and sparse grouped workloads
