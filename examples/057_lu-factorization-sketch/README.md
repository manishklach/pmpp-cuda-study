# 057 - LU Factorization Sketch

- Track: `Linear Algebra`
- Difficulty: `Advanced`
- Status: `Guided template`
- GitHub batch: `041-060`

## Goal

Build and study a library-aware study scaffold of **LU Factorization Sketch**.

## PMPP Ideas To Focus On

- blocked factorization workflow
- panel updates
- library integration planning

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This example is intentionally a stronger study scaffold because it typically depends on CUDA math libraries not available in this authoring environment.
- Use the README plus code comments as a roadmap for the eventual implementation.

## What To Modify Next

- Map the panel/update steps to kernels.
- Compare hand-written pieces with cuSOLVER later.
