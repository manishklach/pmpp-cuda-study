# 056 - Power Iteration

- Track: `Linear Algebra`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `041-060`

## Goal

Build and study a working CUDA implementation of **Power Iteration**.

## PMPP Ideas To Focus On

- repeated matvecs
- vector normalization
- dominant eigenvector estimation

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Validation

- The program prints `PASS` when GPU output matches the CPU reference or stays within tolerance.
- Start with the included tiny matrices before scaling up.

## What To Modify Next

- Track Rayleigh quotient per iteration.
- Try a matrix with a clearer dominant eigenvalue.
