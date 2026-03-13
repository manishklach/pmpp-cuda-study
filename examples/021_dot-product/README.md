# 021 - Dot Product

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Compute a dot product on the GPU using elementwise multiplication plus reduction.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 65536 --block-size 256
```
