# 027 - Prefix Sum Work Efficient Scan

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Compute an inclusive scan using a Blelloch-style work-efficient structure.

## Why This Example Matters

This is the more important scan implementation to understand after the naive version because it reduces redundant work and introduces the classic up-sweep/down-sweep pattern.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 128
```
