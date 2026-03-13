# 053 - Sparse Matrix Dense Vector Multiply

- Track: `Linear Algebra`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `041-060`

## Goal

Build and study a working CUDA implementation of **Sparse Matrix Dense Vector Multiply**.

## PMPP Ideas To Focus On

- sparse-dense interaction
- row traversal
- format awareness

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

- Try a different sparse layout later.
- Increase rows and sparsity.
