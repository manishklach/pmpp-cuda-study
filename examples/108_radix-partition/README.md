# 108 - Radix Partition

## Overview

Partition keys by bit range as a building block for radix sort.

## Why this matters

Radix partitioning is one of the core stable-data-movement patterns on GPU.

## Expected kernel structure

Compute predicates, scan bucket offsets, then scatter keys into bucketed output ranges.

## Future implementation notes

Document stability guarantees and how local offsets become global positions.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
