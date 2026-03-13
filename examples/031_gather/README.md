# 031 - Gather

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `🧪 verified`

## Goal

Read values from irregular source indices and materialize them in a dense output buffer.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 65536 --block-size 256
```
