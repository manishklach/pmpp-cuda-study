# 048 - Sobel Edge Detection

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `041-060`

## Goal

Build and study a working CUDA implementation of **Sobel Edge Detection**.

## PMPP Ideas To Focus On

- paired convolution passes
- gradient magnitude
- edge maps

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

- Threshold the gradient map.
- Compare absolute-sum versus Euclidean magnitude.
