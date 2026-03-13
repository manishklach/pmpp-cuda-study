# 024 - Max Reduction

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Reduce a float array to one maximum value on the GPU and validate it against a CPU reference.

## Why This Example Matters

This is a natural follow-up to sum reduction because the structure is almost identical while the operator changes. That makes it a clean way to reinforce reduction thinking.

## CUDA Concepts Taught

- shared-memory reduction
- operator substitution in collective kernels
- benchmark mode for reduction variants

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 65536 --block-size 256
```

```powershell
.\example.exe --bench --size 1048576 --warmup 5 --iters 20 --block-size 256
```

## Expected Output

- Prints `PASS` when the GPU maximum matches the CPU maximum.

## Correctness Notes

- A known large value is injected into the input to make the expected maximum easy to reason about.

## Likely Bottlenecks

- the same synchronization and occupancy constraints as sum reduction

## Next Optimization Steps

- compare with warp-shuffle reduction
- combine value and index into an argmax variant
