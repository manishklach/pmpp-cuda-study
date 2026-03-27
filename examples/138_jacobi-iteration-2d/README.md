# 138 - Jacobi Iteration 2D

## Overview

Run a 2D Jacobi iteration over a structured grid.

## Why this matters

Jacobi iteration is a clean stepping stone toward iterative PDE solvers.

## Expected kernel structure

Use double buffers, map one thread per grid point, and update from neighbor values only.

## Future implementation notes

Document the role of boundary conditions and convergence checks.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
