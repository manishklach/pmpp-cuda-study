# 030 - Stream Compaction

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Filter positive integers into a compact output buffer on the GPU.

## Why This Example Matters

Compaction is one of the most useful irregular GPU primitives. This baseline uses atomic reservation so the keep/discard pattern is easy to follow.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 128
```
