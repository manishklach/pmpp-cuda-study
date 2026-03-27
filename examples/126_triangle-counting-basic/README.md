# 126 - Triangle Counting Basic

## Overview

Count graph triangles with a basic neighbor-intersection strategy.

## Why this matters

Triangle counting is a canonical irregular graph workload.

## Expected kernel structure

Launch per-edge or per-vertex work, intersect sorted neighbor lists, and accumulate triangle counts carefully.

## Future implementation notes

Start with tiny deterministic graphs and exact CPU validation.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
