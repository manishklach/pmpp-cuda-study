# 094 - Connected Components

- Track: `Graph and ML`
- Difficulty: `Advanced`
- Status: `Guided template`
- GitHub batch: `081-100`

## Goal

Study **Connected Components** in CUDA using a PMPP-style decomposition. Start small, validate correctness, then tune.

## PMPP Ideas To Focus On

- irregular parallelism
- iteration strategy
- scalability planning

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
