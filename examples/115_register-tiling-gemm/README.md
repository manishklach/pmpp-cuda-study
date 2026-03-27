# 115 - Register Tiling Gemm

## Overview

Study a GEMM that uses registers to accumulate multiple outputs per thread.

## Why this matters

Register tiling increases arithmetic intensity beyond simple shared-memory tiling.

## Expected kernel structure

Each thread computes a small output patch, reusing staged data across more fused multiply-adds.

## Future implementation notes

Keep the first implementation shape small and readable instead of chasing peak performance.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
