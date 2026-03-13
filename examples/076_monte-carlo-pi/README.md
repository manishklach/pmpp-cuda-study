# 076 - Monte Carlo Pi

- Track: `Simulation`
- Difficulty: `Intermediate`
- Status: `🧪 verified`

## Goal

Estimate pi with a Monte Carlo kernel and validate the estimate using a tolerance band instead of exact equality.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 8192
```
