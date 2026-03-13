# 007 - SAXPY

- Track: `Foundations`
- Difficulty: `Beginner`
- Status: `Reference-friendly`
- GitHub batch: `001-020`

## Goal

Build and study a working CUDA implementation of **SAXPY**.

## PMPP Ideas To Focus On

- BLAS-style vector ops
- fused arithmetic
- bandwidth-bound kernels

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

- Generalize to z = a*x + b*y.
- Compare different block sizes.
