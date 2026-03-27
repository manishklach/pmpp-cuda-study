# 021 - Dot Product

## Overview

This example computes the dot product of two dense vectors. It includes a simple map-then-sum baseline and a shared-memory block-reduction kernel that is closer to a practical CUDA implementation.

## What this example teaches

- how a dot product combines elementwise parallel work with a reduction
- why block partials are a natural way to scale a reduction
- how to validate a collective GPU result against a CPU reference

## CUDA concepts involved

- 1D grid and block mapping
- coalesced vector loads
- shared-memory block reduction
- synchronization with `__syncthreads()`
- grid-stride loops

## Kernel mapping

- `elementwise_product_kernel`: one thread handles one vector element and writes one product
- `dot_block_reduce_kernel`: each thread accumulates a grid-stride subset, then the block reduces those thread-local sums into one partial
- launch shape: `blocks = ceil(n / block_size)`, `threads = block_size`

## Memory behavior

- both input vectors are read sequentially, so the initial loads are coalesced
- the baseline writes an intermediate products array to global memory
- the reduction kernel keeps partial sums in registers and shared memory, which cuts global-memory traffic

## Correctness approach

- deterministic inputs are generated from a fixed seed
- a CPU reference computes the exact dot product in double precision
- both GPU paths must match the CPU result within `1e-3`

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 65536 --block-size 256
.\example.exe --bench --size 1048576 --warmup 5 --iters 20 --block-size 256
```

## Expected output

- `Validation: PASS`
- benchmark mode reports problem size, timing, throughput, and effective bandwidth

## Common mistakes

- forgetting that reduction kernels usually assume a sensible block size
- comparing a float reduction to a float CPU accumulation and blaming rounding noise on the GPU
- ignoring the extra global-memory traffic of the map-then-sum baseline

## Possible optimizations / next step

- finish the reduction entirely on the GPU with a second reduction pass
- compare against warp-shuffle reductions
- study `023_sum-reduction` next for a more focused reduction discussion
