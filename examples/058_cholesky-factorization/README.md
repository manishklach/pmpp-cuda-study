# 058 - Cholesky Factorization

- Track: `Linear Algebra`
- Difficulty: `Advanced`
- Status: `Guided template`
- GitHub batch: `041-060`

## Goal

Build and study a library-aware study scaffold of **Cholesky Factorization**.

## PMPP Ideas To Focus On

- SPD assumptions
- triangular updates
- factorization validation

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

- Build a tiny SPD matrix and validate on CPU.
- Compare direct code with cuSOLVER later.
