# 114 - Double Buffered Tiling

## Overview

Show how alternating tile buffers can overlap staging and compute.

## Why this matters

Double buffering is a natural next step after basic shared-memory tiling.

## Expected kernel structure

Maintain two tile buffers, prefetch the next tile while computing on the current one, and swap roles each iteration.

## Future implementation notes

Document exactly where lifetime, synchronization, and occupancy tradeoffs appear.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
