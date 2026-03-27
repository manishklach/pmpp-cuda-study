# 132 - Box Blur Separable Optimized

## Overview

Show a separable box blur with an optimized two-pass structure.

## Why this matters

Separable filters are a useful pattern for reducing stencil cost.

## Expected kernel structure

Run a horizontal pass and a vertical pass, reusing local neighborhoods in each direction.

## Future implementation notes

Document the tradeoff between extra passes and cheaper per-pass kernels.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
