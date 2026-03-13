# 028 - Histogram Global Atomics

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Build a histogram with direct global atomics and validate it against a CPU reference.

## Why This Example Matters

This is the baseline irregular primitive for the histogram path. It is intentionally simple and contention-heavy, which makes it the right reference point for later privatized or shared-memory variants.

## CUDA Concepts Taught

- atomics
- irregular writes
- histogram validation

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

- Prints `PASS` when the GPU bin counts match the CPU histogram.

## Next Optimization Steps

- compare with `029_histogram-shared-memory`
- vary the number of bins and input skew to study contention
