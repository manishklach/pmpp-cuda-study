# 119 - Transpose Optimization Ladder

## Overview

Build a sequence of transpose kernels from naive to padded shared-memory versions.

## Why this matters

Optimization ladders help readers connect one improvement to the next.

## Expected kernel structure

Include a naive baseline, coalesced shared-memory variant, and bank-conflict-aware padded version.

## Future implementation notes

Keep performance claims modest and explain each structural change precisely.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
