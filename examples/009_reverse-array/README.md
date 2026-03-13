# 009 - Reverse Array

- Track: `Foundations`
- Difficulty: `Beginner`
- Status: `Reference-friendly`
- GitHub batch: `001-020`

## Goal

Build and study a working CUDA implementation of **Reverse Array**.

## PMPP Ideas To Focus On

- index remapping
- read/write patterns
- easy-to-inspect test data

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

- Try in-place reversal.
- Test odd and even lengths.
