# 068 - FIR Filter

- Track: `Image and Signal`
- Difficulty: `Intermediate`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Apply a short finite impulse response filter to a 1D signal using one output thread per sample.

## PMPP Ideas To Focus On

- convolution windows
- causal tap application
- small constant-size kernels

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This example uses a simple low-pass style tap set.
- The implementation is intentionally direct so the tap indexing is obvious.
- A next optimization is storing the coefficients in constant memory.
