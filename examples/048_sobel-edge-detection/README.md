# 048 - Sobel Edge Detection

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `🧪 verified`

## Goal

Compute Sobel edge magnitude on a simple grayscale image and validate the result against a CPU reference.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 256
```
