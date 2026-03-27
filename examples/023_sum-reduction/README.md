# 023 - Sum Reduction

## Overview

This example reduces a float array to one sum. It includes an interleaved-addressing baseline and a sequential-addressing version so the divergence and synchronization tradeoffs are easy to compare.

## What this example teaches

- how block-level reduction turns many inputs into one partial sum
- why some reduction layouts waste work through divergence
- why reductions almost always rely on shared memory and synchronization

## CUDA concepts involved

- shared memory
- block cooperation
- reduction trees
- control-flow divergence
- host-side final accumulation of block partials

## Kernel mapping

- one block reduces one contiguous chunk of the input
- each thread loads one element into shared memory
- thread 0 of each block writes one partial sum to global memory
- launch shape: `blocks = ceil(n / block_size)`, `threads = block_size`

## Memory behavior

- global loads are coalesced because each block reads a contiguous slice
- shared memory holds the in-block partial sums
- the interleaved baseline creates more divergence as the stride grows
- sequential addressing keeps active threads contiguous and is usually the better teaching baseline

## Correctness approach

- deterministic inputs come from a fixed seed
- a CPU reference uses scalar accumulation in double precision
- both GPU kernels must match the CPU result within `1e-3`

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 65536 --block-size 256
.\example.exe --bench --size 1048576 --warmup 5 --iters 20 --block-size 256
```

## Expected output

- `Validation: PASS`
- benchmark mode reports problem size, timing, element throughput, and effective bandwidth

## Common mistakes

- using a non-power-of-two block size with code that assumes tree-style halving
- removing a `__syncthreads()` and accidentally reading partial sums before a neighbor writes them
- treating the host-side final accumulation as if it were the performance-critical part

## Possible optimizations / next step

- add a second GPU reduction pass
- compare against warp-level reductions
- use this as the bridge into scan and stream-compaction patterns
