# 004 - Scalar Vector Multiply

- Track: `Foundations`
- Difficulty: `Beginner`
- Status: `Reference-friendly`
- GitHub batch: `001-020`

## Goal

Build and study a working CUDA implementation of **Scalar Vector Multiply**.

## PMPP Ideas To Focus On

- kernel parameters
- simple arithmetic throughput
- reference checks

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

- Read the scalar from argv.
- Try doubles if your GPU supports them.
