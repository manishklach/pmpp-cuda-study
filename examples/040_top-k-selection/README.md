# 040 - Top K Selection

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `🧪 verified`

## Goal

Select the largest `k` values by sorting a small array on the GPU and comparing against a CPU top-k reference.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check
```
