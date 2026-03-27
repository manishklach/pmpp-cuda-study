# 105 - Warp-Aggregated Atomics

## Overview

Reduce atomic traffic by aggregating updates at warp granularity.

## What this example teaches

- how warp aggregation differs from naive per-thread atomics
- how ballot masks summarize active lanes inside a warp
- why reducing the number of atomics can matter on hot counters

## CUDA concepts involved

- warp vote intrinsics
- atomics
- warp-level aggregation
- baseline versus improved comparison

## Kernel mapping

- both kernels walk the input in a grid-stride loop
- the naive kernel issues one atomic per positive element
- the warp-aggregated kernel issues at most one atomic per warp per loop iteration

## Memory behavior

- input reads are regular and coalesced
- contention is concentrated on one counter
- the optimization changes atomic traffic, not the input access pattern

## Correctness approach

- deterministic signed input
- CPU reference counts positive values
- both GPU kernels must match exactly

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 65536
.\example.exe --bench --size 1048576 --warmup 5 --iters 20
```

## Expected output

- `Validation: PASS`
- benchmark mode reports problem size, timing, and throughput

## Common mistakes

- treating the warp as if every lane is active when the ballot mask is sparse
- forgetting that the optimization only helps when lanes target the same atomic destination
- assuming warp aggregation removes the atomic bottleneck entirely

## Possible optimizations / next step

- extend the pattern to compaction or histogram bins
- compare with block-private accumulation
- inspect warp occupancy and active-mask behavior for sparse predicates
