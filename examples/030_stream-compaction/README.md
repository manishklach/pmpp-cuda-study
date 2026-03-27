# 030 - Stream Compaction

## Overview

This example filters positive integers into a dense output array. It includes an unordered atomic-reservation baseline and a stable single-block scan-based compactor so the tradeoff between simplicity and structure is explicit.

## What this example teaches

- how stream compaction keeps only selected elements
- why atomic reservation is easy but unstable
- how scan enables stable compaction

## CUDA concepts involved

- atomics
- predicate evaluation
- scan-based index generation
- synchronization-heavy shared-memory coordination

## Kernel mapping

- `compact_atomic_kernel`: threads walk the input and reserve output slots with one global counter
- `compact_stable_scan_kernel`: one block scans the keep flags, then each surviving element writes to its scanned output slot
- launch shape: atomic baseline uses `blocks = ceil(n / 256)`, stable version uses `<<<1, next_power_of_two(n)>>>`

## Memory behavior

- the atomic baseline reads linearly but contends on one global counter
- the stable version keeps the scan in shared memory and preserves order
- the stable version trades less global contention for more synchronization and shared-memory traffic

## Correctness approach

- deterministic signed inputs come from a fixed seed and a repeatable pattern
- a CPU reference keeps only the positive values in original order
- the atomic baseline is validated as a set, while the scan-based version is validated in exact order

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 512
.\example.exe --bench --size 512 --warmup 5 --iters 20
```

## Expected output

- `Validation: PASS`
- benchmark mode reports input size, timing, and element throughput

## Common mistakes

- assuming the atomic baseline preserves input order
- forgetting that stable compaction depends on an exclusive or inclusive scan being interpreted correctly
- scaling this single-block stable implementation to large arrays without adding a hierarchical design

## Possible optimizations / next step

- extend the stable path to a multi-block compaction pipeline
- benchmark atomic versus scan-based compaction on different keep ratios
- reuse the scan building block in sorting or radix-based workflows
