# 052 - Sparse Matrix Vector Multiply CSR

- Track: `Linear Algebra`
- Difficulty: `Advanced`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Multiply a sparse matrix in CSR format by a dense vector and validate the output against a CPU reference.

## Why This Example Matters

CSR SpMV is one of the most important sparse GPU kernels. It introduces irregular memory access and makes the repo’s linear-algebra path feel much more complete.

## CUDA Concepts Taught

- CSR traversal
- row-parallel sparse kernels
- sparse validation and benchmarking

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 128
```

```powershell
.\example.exe --bench --size 4096 --warmup 5 --iters 20
```

## Expected Output

- Prints `PASS` when the GPU vector matches the CPU reference within tolerance.

## Next Optimization Steps

- compare different sparse formats
- experiment with row imbalance and larger nonzero counts
