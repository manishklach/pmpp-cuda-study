# 106 - Block Private Accumulation

## Overview

Accumulate per-block state privately before flushing global results.

## Why this matters

Block-private accumulation is a standard way to reduce contention.

## Expected kernel structure

Each block builds a private summary in shared memory and flushes a smaller set of global updates.

## Future implementation notes

Compare this with direct atomics and track where synchronization becomes necessary.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
