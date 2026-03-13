# 011 - Threshold Binary Mask

- Track: `Foundations`
- Difficulty: `Beginner`
- Status: `Reference-friendly`
- GitHub batch: `001-020`

## Goal

Build and study a working CUDA implementation of **Threshold Binary Mask**.

## PMPP Ideas To Focus On

- predicate kernels
- binary outputs
- simple segmentation

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

- Emit 0/255 instead of 0/1.
- Add lower and upper thresholds.
