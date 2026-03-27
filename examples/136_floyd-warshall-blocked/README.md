# 136 - Floyd Warshall Blocked

## Overview

Explore a blocked all-pairs shortest-path update.

## Why this matters

Blocked Floyd-Warshall is a compact study in dependency-aware tiling.

## Expected kernel structure

Iterate over pivot tiles and update dependent blocks in phased kernels or staged loops.

## Future implementation notes

Keep the graph tiny at first so the dependency structure stays readable.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
