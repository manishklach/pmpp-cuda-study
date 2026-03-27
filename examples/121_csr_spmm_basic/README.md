# 121 - CSR_SpMM_Basic

## Overview

Multiply a CSR sparse matrix by a dense matrix.

## Why this matters

SpMM is a practical sparse-kernel bridge beyond basic SpMV.

## Expected kernel structure

Map rows to thread groups, iterate CSR ranges, and accumulate multiple output columns.

## Future implementation notes

Start with small dense widths and a CPU reference before exploring better row scheduling.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
