# 020 - Matrix Transpose With Shared Memory

- Track: `Foundations`
- Difficulty: `Intermediate`
- Status: `✅ fully mature`
- Maturity: `Level 6 - polished teaching example`

## Goal

Transpose a matrix using a padded shared-memory tile and validate the result against a CPU reference.

## Why This Example Matters

Transpose is one of the clearest ways to study memory layout. This version is especially useful because it shows how shared memory and padding help organize access cleanly.

## CUDA Concepts Taught

- 2D thread mapping
- shared-memory tiling
- padded tiles
- transpose access patterns

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 64
```

```powershell
.\example.exe --bench --size 256 --warmup 5 --iters 20
```

## Expected Output

- Prints `PASS` when the transposed GPU output matches the CPU reference.

## Next Optimization Steps

- compare with `019_matrix-transpose-naive`
- study how tile padding affects shared-memory access behavior
