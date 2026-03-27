# 029 - Histogram Shared Memory

## Overview

This example adds per-block shared-memory privatization to the histogram path. It validates both the global-atomic baseline and the shared-memory version so the optimization step stays grounded in correctness.

## What this example teaches

- how privatization reduces global-atomic contention
- why shared memory is a good staging area for block-local histograms
- how to compare a baseline and an improved kernel on the same input

## CUDA concepts involved

- shared-memory privatization
- block-local accumulation
- global atomic flush
- contention tradeoffs

## Kernel mapping

- both kernels use a grid-stride loop over the input
- the baseline atomically updates global bins directly
- the privatized kernel accumulates into per-block shared bins, then flushes those totals globally
- launch shape: `blocks = ceil(n / block_size)`, `threads = block_size`

## Memory behavior

- input reads are still linear and coalesced
- shared-memory atomics are usually cheaper than fighting over the same global bins
- the final flush still uses global atomics, but now it happens once per block per bin instead of once per sample

## Correctness approach

- deterministic hot-bin input is shared with the baseline histogram example
- a CPU reference builds the expected bin counts
- both GPU kernels must match the CPU histogram exactly

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 65536 --block-size 256
.\example.exe --bench --size 1048576 --warmup 5 --iters 20 --block-size 256
```

## Expected output

- `Validation: PASS`
- benchmark mode reports sample count, timing, throughput, and effective bandwidth

## Common mistakes

- forgetting to initialize the shared histogram before accumulation
- omitting the synchronization before flushing shared bins to global memory
- assuming shared-memory privatization removes contention completely instead of moving most of it off the global path

## Possible optimizations / next step

- compare timings against `028_histogram-global-atomics`
- try skewed versus uniform inputs to study contention tradeoffs
- extend the design with per-warp privatization or larger bin counts
