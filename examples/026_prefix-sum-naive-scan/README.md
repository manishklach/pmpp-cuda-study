# 026 - Prefix Sum Naive Scan

## Overview

This example implements an inclusive Hillis-Steele scan in a single block. It is intentionally not work-efficient, but it makes the scan data dependencies and synchronization requirements very easy to inspect.

## What this example teaches

- how inclusive scan builds each prefix total from earlier elements
- why a naive scan performs extra work
- why scan kernels need careful synchronization between phases

## CUDA concepts involved

- shared memory
- block-wide synchronization
- divergence from offset-based participation
- inclusive scan

## Kernel mapping

- one block handles the full input
- one thread maps to one input element
- each offset step updates all valid threads whose index is at least that offset
- launch shape: `<<<1, n>>>`

## Memory behavior

- the input is loaded once into shared memory
- each phase rereads earlier shared-memory values and writes updated prefix totals back
- every offset doubles the look-back distance, so the kernel does `O(n log n)` shared-memory additions

## Correctness approach

- deterministic integer inputs come from a fixed seed
- a CPU reference computes the inclusive scan sequentially
- the GPU output must match the CPU output exactly

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 256
.\example.exe --bench --size 512 --warmup 5 --iters 20
```

## Expected output

- `Validation: PASS`
- benchmark mode reports scanned element count and timing

## Common mistakes

- forgetting the extra synchronization needed to keep one scan phase from reading values written by the same phase
- assuming this algorithm is efficient just because it is parallel
- trying to scale this single-block teaching kernel to very large inputs without a multi-block design

## Possible optimizations / next step

- compare directly with `027_prefix-sum-work-efficient-scan`
- extend the scan to a two-level multi-block implementation
- use scan as the backbone for stable stream compaction
