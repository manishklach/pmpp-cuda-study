# 139 - LBM D2Q9 Demo

## Overview

Study one D2Q9 lattice-Boltzmann update step.

## Why this matters

LBM is a compact systems-style stencil plus streaming workload.

## Expected kernel structure

Track nine distributions per cell, collide locally, then stream to neighboring cells.

## Future implementation notes

Start with a tiny grid and emphasize layout choices before physics depth.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
