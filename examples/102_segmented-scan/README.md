# 102 - Segmented Scan

## Overview

Compute prefix sums that reset at segment boundaries.

## What this example teaches

- how segmented scan differs from a plain scan
- why head flags need to travel with the partial sums
- where synchronization cost appears in a naive segmented implementation

## CUDA concepts involved

- shared memory
- head-flag propagation
- synchronization-heavy Hillis-Steele style scan

## Kernel mapping

- one block processes the full study input
- one thread owns one logical element
- each scan step checks whether the segment boundary blocks accumulation from the left

## Memory behavior

- data is staged in shared memory once
- each pass rereads and rewrites shared state
- this version is intentionally more about clarity than about minimizing work

## Correctness approach

- deterministic integer input and deterministic segment heads
- CPU reference resets the running sum whenever a head flag is set
- GPU output must match exactly

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 128
.\example.exe --bench --size 256 --warmup 5 --iters 20
```

## Expected output

- `Validation: PASS`
- benchmark mode reports scanned element count and timing

## Common mistakes

- letting additions cross a segment boundary
- forgetting that all threads must observe the same head-flag updates between steps
- assuming this one-block study version scales directly to long arrays

## Possible optimizations / next step

- move to a work-efficient segmented scan
- extend the design to multi-block segments
- connect the result to segmented compaction or sparse row processing
