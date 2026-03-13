# 029 - Histogram Shared Memory

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Build a histogram using per-block shared-memory accumulation before flushing bins globally.

## Why This Example Matters

This is the first real optimization step after the global-atomic baseline. It shows how privatization reduces contention while preserving correctness.

## CUDA Concepts Taught

- shared-memory privatization
- reduced atomic contention
- baseline vs optimized histogram thinking

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 65536
```

```powershell
.\example.exe --bench --size 1048576 --warmup 5 --iters 20
```

## Expected Output

- Prints `PASS` when the GPU bins match the CPU histogram exactly.

## Next Optimization Steps

- compare runtime directly against `028_histogram-global-atomics`
- try skewed inputs to see contention effects more clearly
