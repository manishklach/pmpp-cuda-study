# 113 - CpAsync Style Tiled Matmul Demo

## Overview

Study the structure of an asynchronous-tiling GEMM pipeline without requiring hardware-specific features.

## Why this matters

Even a demo helps explain why load-compute overlap matters in modern CUDA kernels.

## Expected kernel structure

Use staged tile loads, software pipelining ideas, and comments that map to cp.async-style behavior.

## Future implementation notes

Be explicit about what is conceptual versus what is hardware-backed in a future implementation.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
