# 110 - Counting Sort Parallel

## Overview

Use histogramming plus prefix sums to place keys into sorted order.

## Why this matters

Counting sort demonstrates how histogram and scan compose into a complete pipeline.

## Expected kernel structure

Build counts, scan bucket starts, then scatter keys into deterministic output positions.

## Future implementation notes

Start with small key ranges and verify exact ordering behavior.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
