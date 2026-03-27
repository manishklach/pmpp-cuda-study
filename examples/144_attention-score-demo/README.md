# 144 - Attention Score Demo

## Overview

Compute small attention score matrices from query and key blocks.

## Why this matters

Attention-score kernels connect GEMM-style thinking to transformer workloads.

## Expected kernel structure

Map tiles of Q and K to blocks, accumulate score tiles, then optionally apply scaling.

## Future implementation notes

Keep dimensions tiny and deterministic to preserve readability.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
