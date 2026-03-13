# 034 - Find First Match

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `🧪 verified`

## Goal

Find the smallest index containing a target value using a parallel atomic-min strategy.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 65536 --block-size 256
```
