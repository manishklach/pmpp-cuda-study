# 109 - Select Topk Basic

## Overview

Build a simple top-k selection study kernel before heap or block-select optimizations.

## Why this matters

Top-k selection connects reduction patterns to practical ranking workloads.

## Expected kernel structure

Use staged partial selection per block and merge candidates afterward.

## Future implementation notes

Keep the first implementation correctness-focused and explicit about tradeoffs.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
