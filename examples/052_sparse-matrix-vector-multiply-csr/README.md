# 052 - Sparse Matrix Vector Multiply CSR

- Track: `Linear Algebra`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `041-060`

## Goal

Build and study a working CUDA implementation of **Sparse Matrix Vector Multiply CSR**.

## PMPP Ideas To Focus On

- CSR layout
- one-row-per-thread mapping
- irregular memory access

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

- Change sparsity patterns.
- Compare against a dense fallback.
