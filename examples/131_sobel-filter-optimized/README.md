# 131 - Sobel Filter Optimized

## Overview

Apply a Sobel edge filter with tiled shared-memory staging.

## What this example teaches

- how image kernels use halo tiles
- why small convolution-style operators benefit from shared-memory reuse
- how to validate a tiled image-processing kernel against a CPU reference

## CUDA concepts involved

- 2D thread mapping
- halo loading
- shared-memory reuse
- Sobel gradient evaluation

## Kernel mapping

- each block covers a 16x16 output tile
- border threads pull in halo pixels around the tile
- every output pixel computes Sobel X and Sobel Y from the staged neighborhood

## Memory behavior

- each pixel neighborhood is loaded once into shared memory
- repeated neighbor accesses then come from shared memory
- halo loads and the synchronization barrier are the main structural overheads

## Correctness approach

- deterministic synthetic grayscale image
- CPU reference applies the same Sobel stencil
- GPU output must match within `1e-5`

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 128
.\example.exe --bench --size 512 --warmup 5 --iters 20
```

## Expected output

- `Validation: PASS`
- benchmark mode reports pixel count, timing, and throughput

## Common mistakes

- forgetting corner halo loads
- mixing Sobel X and Sobel Y coefficient placement
- benchmarking before confirming the tiled kernel matches the CPU reference exactly

## Possible optimizations / next step

- compare against a direct global-memory Sobel baseline
- add vectorized loads for wider images
- connect the halo logic to the generic stencil and heat-diffusion examples
