# 120 - Stencil With Halo Tiling

## Overview

Stage a 2D stencil tile plus halo cells in shared memory.

## What this example teaches

- how halo loads differ from the simple interior tile
- why shared-memory staging helps stencil kernels reuse neighboring values
- how to keep boundary handling explicit without hiding the stencil structure

## CUDA concepts involved

- 2D blocks and grids
- halo tiling
- shared-memory staging
- synchronization before stencil evaluation

## Kernel mapping

- each block covers a 16x16 output tile
- every thread loads its center cell
- border threads cooperatively load halo cells before the stencil update runs

## Memory behavior

- the tile and its halo are loaded once from global memory
- neighbor reads then come from shared memory
- the main cost is the extra halo traffic and the synchronization barrier before compute

## Correctness approach

- deterministic float grid input
- CPU reference applies the same five-point weighted stencil
- GPU output must match within `1e-5`

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 128
.\example.exe --bench --size 512 --warmup 5 --iters 20
```

## Expected output

- `Validation: PASS`
- benchmark mode reports grid size, timing, and throughput

## Common mistakes

- forgetting to load halo cells for edge threads
- letting some threads skip the synchronization barrier while others continue
- confusing clamped boundary behavior with halo staging logic

## Possible optimizations / next step

- extend to wider stencils or multiple time steps
- compare against a direct global-memory stencil baseline
- connect the halo pattern to Sobel and heat-diffusion examples
