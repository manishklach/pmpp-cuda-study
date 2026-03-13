# 037 - Odd Even Merge Sort

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `021-040`

## Goal

Build and study a working CUDA implementation of **Odd Even Merge Sort**.

## PMPP Ideas To Focus On

- merge network structure
- compare-exchange pairs
- network-style sorting

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Validation

- The program prints `PASS` when GPU output matches the CPU reference.
- These examples use intentionally small inputs so each pattern is easy to inspect first.

## What To Modify Next

- Visualize the stages on paper.
- Compare its output and cost with bitonic sort.
