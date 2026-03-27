# 028 - Histogram Global Atomics

## Overview

This example builds a small histogram by updating global-memory bins with `atomicAdd`. It is the simplest correct CUDA histogram, and it serves as the baseline for the shared-memory privatization example that follows.

## What this example teaches

- how to map one input element stream into a fixed set of bins
- why irregular write patterns often require atomics
- why contention can dominate histogram performance

## CUDA concepts involved

- global atomics
- grid-stride loops
- irregular write patterns
- exact integer validation

## Kernel mapping

- threads walk the input in a grid-stride loop
- each thread hashes its sample to a bin and performs one atomic increment
- launch shape: `blocks = ceil(n / block_size)`, `threads = block_size`

## Memory behavior

- input reads are linear and coalesced
- bin updates are irregular and may serialize heavily when many samples target the same bin
- this example intentionally uses hot bins so the contention cost is easy to discuss

## Correctness approach

- deterministic input values are generated from a fixed seed and a repeatable hot-bin pattern
- a CPU reference builds the same histogram on the host
- the GPU bin counts must match exactly

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

- assuming coalesced reads guarantee good performance even when the writes are highly contended
- forgetting to zero the output bins before each run
- using this baseline to judge histograms without also looking at contention patterns

## Possible optimizations / next step

- compare directly with `029_histogram-shared-memory`
- increase the number of bins or change the skew to study how contention moves
- experiment with per-warp or per-block privatization
