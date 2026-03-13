# 032 - Scatter

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `🧪 verified`

## Goal

Write dense input values into irregular destination indices on the GPU.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 65536 --block-size 256
```
