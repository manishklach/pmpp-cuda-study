# 125 - Sssp_Delta Stepping Simplified

## Overview

Study a simplified bucket-based shortest-path iteration.

## Why this matters

Bucketed SSSP shows how irregular frontiers and relaxation interact on GPU.

## Expected kernel structure

Group vertices into distance buckets, process relaxations, and update active worklists.

## Future implementation notes

Be explicit about the simplifications compared with production delta-stepping.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
