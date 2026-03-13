# 046 - Convolution 2D

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `🧪 verified`

## Goal

Apply a direct 3x3 2D convolution on the GPU and validate the image against a CPU reference.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 256
```
