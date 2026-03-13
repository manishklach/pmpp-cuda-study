# 041 - Matrix Vector Multiply

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `🧪 verified`

## Goal

Multiply a dense matrix by a dense vector and validate the result against a CPU reference.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 1024 --block-size 256
```
