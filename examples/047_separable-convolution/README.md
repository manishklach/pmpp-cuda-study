# 047 - Separable Convolution

- Track: `Linear Algebra`
- Difficulty: `Intermediate`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Apply a separable 3-tap blur with horizontal and vertical GPU passes and validate against a CPU reference.

## Why This Example Matters

This is a clean optimization-oriented image kernel because it turns one 2D stencil into two simpler 1D passes.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 64
```
