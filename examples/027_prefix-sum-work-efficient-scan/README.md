# 027 - Prefix Sum Work-Efficient Scan

## Overview

This example implements an inclusive Blelloch-style scan using an up-sweep and down-sweep in shared memory. Compared with the naive version, it performs only `O(n)` total additions instead of `O(n log n)`.

## What this example teaches

- how the Blelloch scan builds and then distributes prefix totals
- why work-efficient scan differs from naive scan
- how synchronization boundaries line up with tree phases

## CUDA concepts involved

- shared memory
- up-sweep / down-sweep tree traversal
- synchronization after every tree level
- padded power-of-two launches

## Kernel mapping

- one block handles the full input, padded to the next power of two
- one thread maps to one input slot in the padded array
- up-sweep forms subtree totals, then down-sweep converts them into prefix values
- launch shape: `<<<1, next_power_of_two(n)>>>`

## Memory behavior

- the working set lives in shared memory after the initial load
- the tree structure reduces redundant additions compared with Hillis-Steele
- later phases involve fewer active threads, so divergence is present but the total work is lower

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

- forgetting that this block-level implementation pads to a power of two
- mixing up exclusive and inclusive scan during the final writeback
- removing synchronization between tree levels and corrupting the prefix totals

## Possible optimizations / next step

- compare the timing and explanation directly against `026_prefix-sum-naive-scan`
- extend the algorithm to hierarchical multi-block scan
- reuse this scan pattern in stream compaction or radix-sort building blocks
