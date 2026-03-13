# 012 - RGB To Grayscale

- Track: `Foundations`
- Difficulty: `Beginner`
- Status: `Reference-friendly`
- GitHub batch: `001-020`

## Goal

Build and study a working CUDA implementation of **RGB To Grayscale**.

## PMPP Ideas To Focus On

- pixel structs
- image indexing
- weighted color transforms

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

- Try alternative luminance weights.
- Preserve alpha in a uchar4 variant.
