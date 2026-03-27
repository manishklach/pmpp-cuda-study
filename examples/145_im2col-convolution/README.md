# 145 - Im2Col Convolution

## Overview

Lower image patches into columns as a convolution study path.

## Why this matters

im2col is a practical systems trick that trades memory for simpler GEMM reuse.

## Expected kernel structure

Extract sliding-window patches into a matrix, then multiply by filters or inspect the lowered layout.

## Future implementation notes

Explain the memory expansion clearly and keep the first case small.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
