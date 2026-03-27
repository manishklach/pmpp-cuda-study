# 117 - Coalescing Study

## Overview

Contrast a coalesced copy with a deliberately strided copy pattern.

## What this example teaches

- what coalesced access means in practice
- why neighboring threads should usually read neighboring addresses
- how to keep a memory-pattern study honest with exact output checks

## CUDA concepts involved

- coalesced global-memory access
- strided loads
- baseline versus contrastive access pattern study

## Kernel mapping

- both kernels map one thread to one logical output element
- the coalesced kernel reads contiguous values
- the strided kernel reads every 32nd value from a larger backing array

## Memory behavior

- the coalesced kernel matches the GPU's preferred access pattern
- the strided kernel keeps the arithmetic identical but makes neighboring threads touch distant addresses
- the example isolates access pattern as the main difference

## Correctness approach

- deterministic float backing array
- CPU references for both the contiguous and strided outputs
- both GPU kernels must match their expected results exactly

## Build and run

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 4096
.\example.exe --bench --size 65536 --warmup 5 --iters 20
```

## Expected output

- `Validation: PASS`
- benchmark mode reports logical output size, timing, and throughput

## Common mistakes

- comparing kernels that do different work and calling it a coalescing study
- hiding the stride inside the indexing formula without explaining it
- reading benchmark numbers before confirming both outputs are correct

## Possible optimizations / next step

- compare different strides instead of only one fixed stride
- fold the access study into a transpose or gather/scatter example
- pair the study with cache-behavior notes if you later deepen the topic
