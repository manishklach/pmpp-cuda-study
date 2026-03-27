# 141 LayerNorm Forward

## Overview

This example implements a forward-pass layer normalization kernel over a small batch of feature vectors. Each thread block owns one row, cooperatively reduces the row sum and squared sum in shared memory, then normalizes and applies affine scale and bias.

## What this example teaches

- how a reduction-style kernel appears inside a practical ML primitive
- why one block per row is a natural mapping when the hidden dimension fits in a block
- how mean and variance become shared statistics reused by every thread in the row

## CUDA concepts involved

- block-level reduction
- shared-memory staging
- synchronization with `__syncthreads()`
- row-wise normalization and elementwise affine transform

## Kernel mapping

- grid: one block per row
- block: `256` threads, one thread per feature value
- each thread reads one input value, participates in the statistics reduction, then writes one normalized output

## Memory behavior

Input, output, gamma, and beta are all read from global memory. The expensive part is not arithmetic; it is reusing the row statistics. Shared memory lets the block compute mean and variance once, then broadcast them to all threads in the row.

## Correctness approach

A CPU reference computes the same mean, variance, normalization, and affine transform row by row. The example copies the GPU output back, compares with a floating-point tolerance, and prints `PASS` or `FAIL`.

## Build and run

```powershell
cd examples\141_layernorm-forward
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 4096
.\example.exe --bench --size 65536 --warmup 5 --iters 20
```

## Expected output

```text
Example: 141_layernorm-forward
Mode: check
Validation: PASS
```

## Common mistakes

- forgetting that all threads need the reduced sums before normalization starts
- using the wrong denominator when computing variance
- assuming layer norm reduces across the batch instead of across each row

## Possible optimizations / next step

A natural next step is to replace the full shared-memory reduction with warp shuffles for smaller hidden sizes, or to fuse the affine transform with surrounding operators in a larger inference pipeline.
