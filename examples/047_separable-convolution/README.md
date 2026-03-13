# 047 - Separable Convolution

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `041-060`

## Goal

Build and study a working CUDA implementation of **Separable Convolution**.

## PMPP Ideas To Focus On

- two-pass filtering
- algorithmic speedup
- intermediate buffers

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

- Compare against direct 2D convolution.
- Use different horizontal and vertical kernels.
