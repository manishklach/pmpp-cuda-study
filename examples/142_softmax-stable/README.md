# 142 Softmax Stable

## Overview

This example implements a numerically stable row-wise softmax. Each block handles one row of logits, first reduces the row maximum, then subtracts that maximum before exponentiation so large logits do not overflow.

## What this example teaches

- why stable softmax is a reduction-plus-normalization pattern
- how block-level reductions can support a practical ML primitive
- why numerical stability matters even in otherwise simple kernels

## CUDA concepts involved

- max reduction in shared memory
- sum reduction in shared memory
- synchronization between reduction phases
- row-wise normalization

## Kernel mapping

- grid: one block per row of logits
- block: `128` threads, one thread per class value
- the same block computes row max, denominator, and final probabilities

## Memory behavior

The kernel reads each logit once, stages intermediate reduction values in shared memory, and writes one probability per element. Shared memory is used to reuse the row maximum and denominator instead of recomputing them for each thread.

## Correctness approach

A CPU reference performs the same stable softmax calculation row by row. The example compares the GPU probabilities with a floating-point tolerance and prints `PASS` or `FAIL`.

## Build and run

```powershell
cd examples\142_softmax-stable
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 4096
.\example.exe --bench --size 65536 --warmup 5 --iters 20
```

## Expected output

```text
Example: 142_softmax-stable
Mode: check
Validation: PASS
```

## Common mistakes

- exponentiating raw logits without subtracting the maximum first
- forgetting to synchronize between the max and sum reductions
- treating softmax as a single global reduction instead of one reduction per row

## Possible optimizations / next step

A useful next step is a warp-specialized version for shorter rows, followed by a fused scale-mask-softmax variant like the one used in attention kernels.
