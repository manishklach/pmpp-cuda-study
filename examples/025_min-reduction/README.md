# 025 - Min Reduction

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Reduce a float array to one minimum value on the GPU and validate it against a CPU reference.

## Why This Example Matters

This completes the trio of basic reduction operators in the trusted core and reinforces that the collective structure matters more than the specific operator.

## CUDA Concepts Taught

- shared-memory reduction
- reduction operator variants
- CPU reference validation

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

- Prints `PASS` when the GPU minimum matches the CPU minimum.

## Correctness Notes

- A known low outlier is injected into the input for easy reasoning.

## Likely Bottlenecks

- synchronization and memory traffic costs similar to the max-reduction kernel

## Next Optimization Steps

- compare max, min, and sum throughput under the same settings
- promote to an argmin variant that tracks the index as well
