# 129 - Spatial Hashing Nearest Cell

## Overview

Map points into hashed spatial cells as a prelude to local-neighborhood search.

## Why this matters

Spatial hashing is common in simulation and particle systems.

## Expected kernel structure

Hash each point to a cell, sort or bucket by cell, then query nearby cells for candidates.

## Future implementation notes

Start with deterministic 2D points and emphasize the data-layout story.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
