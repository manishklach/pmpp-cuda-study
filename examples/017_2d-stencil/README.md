# 017 - 2D Stencil

- Track: `Foundations`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `001-020`

## Goal

Build and study a working CUDA implementation of **2D Stencil**.

## PMPP Ideas To Focus On

- 2D indexing
- halo boundaries
- grid launch geometry

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Validation

- The program prints `PASS` when GPU output matches the CPU reference or expected pattern.
- Start with the built-in small inputs before scaling up.

## What To Modify Next

- Swap box average for Laplacian.
- Add shared-memory tiling later.
