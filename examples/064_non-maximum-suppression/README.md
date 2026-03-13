# 064 - Non Maximum Suppression

- Track: `Image and Signal`
- Difficulty: `Advanced`
- Status: `Guided template`
- GitHub batch: `061-080`

## Goal

Study **Non Maximum Suppression** in CUDA using a PMPP-style decomposition. Start small, validate correctness, then tune.

## PMPP Ideas To Focus On

- 2D or chunk indexing
- boundary handling
- pipeline composition

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
