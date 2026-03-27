# 116 - Bank Conflict Study

## Overview

Compare a shared-memory transpose tile that can trigger bank conflicts with a padded version that avoids the common conflict pattern.

## What this example teaches

- what a bank-conflict-prone shared-memory access looks like
- why one extra column of padding can matter
- how to separate correctness from performance in a memory-layout study

## CUDA concepts involved

- shared memory
- transpose-style access
- bank conflicts
- padding for conflict avoidance

## Kernel mapping

- blocks cover 32x32 matrix tiles
- each thread loads one input element and later writes one transposed output element
- the only structural change between kernels is the shared-memory tile shape

## Memory behavior

- global-memory traffic is the same in both kernels
- the unpadded tile can map transposed accesses onto the same shared-memory banks
- the padded tile changes the addressing pattern while preserving the result

## Correctness approach

- deterministic float matrix input
- CPU reference computes a plain transpose
- both GPU kernels must match the CPU result

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 128
.\example.exe --bench --size 512 --warmup 5 --iters 20
```

## Expected output

- `Validation: PASS`
- benchmark mode reports matrix size, timing, and throughput

## Common mistakes

- changing the memory layout and accidentally changing the transpose indexing too
- assuming the bank-conflict fix changes correctness rather than only access behavior
- comparing timings without confirming both kernels produce identical output

## Possible optimizations / next step

- compare directly against earlier transpose examples in the core track
- vary tile sizes and bank-conflict patterns
- extend the study into a transpose optimization ladder
