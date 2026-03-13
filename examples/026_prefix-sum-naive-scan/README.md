# 026 - Prefix Sum Naive Scan

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Compute an inclusive scan using a simple Hillis-Steele style one-block implementation.

## Why This Example Matters

This is the conceptual baseline for scan. It is not the most efficient form, but it makes the data dependencies easy to see.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 128
```
