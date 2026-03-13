# 022 - L2 Norm

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Compute the L2 norm of a vector by reducing the sum of squared elements on the GPU.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 65536 --block-size 256
```
