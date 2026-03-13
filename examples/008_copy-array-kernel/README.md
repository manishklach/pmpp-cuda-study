# 008 - Copy Array Kernel

- Track: `Foundations`
- Difficulty: `Beginner`
- Status: `Reference-friendly`
- GitHub batch: `001-020`

## Goal

Build and study a working CUDA implementation of **Copy Array Kernel**.

## PMPP Ideas To Focus On

- memory movement
- minimal kernels
- launch overhead

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

- Compare with cudaMemcpy when CUDA is available.
- Use float4 or uchar4 data.
