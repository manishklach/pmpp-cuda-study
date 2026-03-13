# 050 - Median Filter

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `🧪 verified`

## Goal

Apply a 3x3 median filter on the GPU and validate it against a CPU reference.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 256
```
