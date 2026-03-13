# 053 - Sparse Matrix Dense Vector Multiply

- Track: `Linear Algebra`
- Difficulty: `Advanced`
- Status: `Guided template`
- GitHub batch: `041-060`

## Goal

Study **Sparse Matrix Dense Vector Multiply** in CUDA using a PMPP-style decomposition. Start small, validate correctness, then tune.

## PMPP Ideas To Focus On

- data layout
- memory reuse
- correctness against a CPU reference

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Checklist

- Describe the parallel unit of work.
- Explain the launch configuration.
- Compare GPU output against a CPU reference.
- Note one correctness risk and one performance risk.
- Write one extension you want to try next.
