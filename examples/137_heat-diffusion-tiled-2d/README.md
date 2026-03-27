# 137 - Heat Diffusion Tiled 2D

## Overview

Compute one tiled 2D heat-diffusion step with halo staging.

## What this example teaches

- how a PDE-style update maps onto a shared-memory tile
- why halo loads are central to 2D grid solvers
- how to keep a physics-style update readable and verifiable

## CUDA concepts involved

- 2D tiling
- halo staging
- five-point Laplacian update
- correctness-first simulation stepping

## Kernel mapping

- each block updates one 16x16 output tile
- threads load interior points and border neighbors into shared memory
- each output point uses the staged tile to compute one diffusion step

## Memory behavior

- neighbor values are reused from shared memory after the halo load
- global traffic is reduced compared with rereading every neighbor directly
- synchronization is required before computing the stencil update

## Correctness approach

- deterministic float grid input
- CPU reference computes the same single diffusion step
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

- forgetting that halo loads are part of the kernel, not optional setup work
- confusing the diffusion coefficient with the stencil weights
- benchmarking the kernel without first verifying the boundary behavior

## Possible optimizations / next step

- apply multiple time steps in a loop or ping-pong buffer sequence
- compare against a direct global-memory stencil
- extend the study to Jacobi or lattice-Boltzmann style updates
