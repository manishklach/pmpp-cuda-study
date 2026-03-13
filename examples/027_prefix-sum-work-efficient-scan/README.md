# 027 - Prefix Sum Work Efficient Scan

- Track: `Parallel Patterns`
- Difficulty: `Intermediate`
- Status: `Guided template`
- GitHub batch: `021-040`

## Goal

Study **Prefix Sum Work Efficient Scan** in CUDA using a PMPP-style decomposition. Start small, validate correctness, then tune.

## PMPP Ideas To Focus On

- work decomposition
- shared memory or atomics
- validation before tuning

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
