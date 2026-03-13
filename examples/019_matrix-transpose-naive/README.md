# 019 - Matrix Transpose Naive

- Track: `Foundations`
- Difficulty: `Intermediate`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Transpose a matrix using a direct global-memory kernel and validate the result against a CPU reference.

## Why This Example Matters

This is the baseline for studying transpose memory behavior before shared-memory tiling is introduced.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 64
```

```powershell
.\example.exe --bench --size 256 --warmup 5 --iters 20
```
