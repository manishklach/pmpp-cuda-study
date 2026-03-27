# 124 - Bfs_Frontier Expansion

## Overview

Expand a graph frontier in parallel as a basic BFS building block.

## Why this matters

Frontier processing is the core of many graph traversal kernels.

## Expected kernel structure

Map edges of the active frontier to threads, mark next-level vertices, and compact the next frontier.

## Future implementation notes

Keep graph storage small and deterministic so correctness is easy to inspect.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
