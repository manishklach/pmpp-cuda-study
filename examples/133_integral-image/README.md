# 133 - Integral Image

## Overview

Build an integral image suitable for fast rectangular region sums.

## Why this matters

Integral images connect scan ideas to image-processing pipelines.

## Expected kernel structure

Compute row prefixes, then column prefixes, while preserving exact integer correctness.

## Future implementation notes

Keep the first implementation simple and clear about data dependencies.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
