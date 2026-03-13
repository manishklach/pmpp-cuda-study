# 019 - Matrix Transpose Naive

- Track: `Foundations`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `001-020`

## Goal

Build and study a working CUDA implementation of **Matrix Transpose Naive**.

## PMPP Ideas To Focus On

- row-major indexing
- strided writes
- baseline transpose

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

- Try rectangular inputs.
- Time it against the tiled version later.
