# 093 - PageRank

- Track: `Graph and ML`
- Difficulty: `Advanced`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Run several PageRank iterations on a deterministic graph and validate the ranks against a CPU reference.

## Why This Example Matters

PageRank is a strong graph-analytics study example because it combines sparse-style graph traversal with iterative floating-point updates.

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 16
```
