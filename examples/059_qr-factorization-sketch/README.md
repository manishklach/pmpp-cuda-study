# 059 - QR Factorization Sketch

- Track: `Linear Algebra`
- Difficulty: `Advanced`
- Status: `Guided template`
- GitHub batch: `041-060`

## Goal

Build and study a library-aware study scaffold of **QR Factorization Sketch**.

## PMPP Ideas To Focus On

- Householder reflections
- orthogonalization workflow
- library handoff points

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

- Derive the CPU reference first.
- Identify which parts belong in cuSOLVER.
