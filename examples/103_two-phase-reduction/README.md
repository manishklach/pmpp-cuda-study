# 103 - Two Phase Reduction

## Overview

Stage a large reduction through block partials and a second aggregation phase.

## Why this matters

Two-phase reduction is the practical form of large-array aggregation on GPU.

## Expected kernel structure

First kernel writes block partials, second kernel reduces those partials, host validates the final scalar.

## Future implementation notes

Compare this structure with single-pass reductions and study when a second kernel is worth it.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
