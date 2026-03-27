# 118 - Shared Memory Staging Patterns

## Overview

Catalog a few common staging patterns for loading data into shared memory.

## Why this matters

Many optimized kernels differ more in staging pattern than in arithmetic.

## Expected kernel structure

Compare direct tile loads, padded tiles, and halo staging with explicit comments around synchronization.

## Future implementation notes

Keep the code small and focus on pattern literacy rather than breadth.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
