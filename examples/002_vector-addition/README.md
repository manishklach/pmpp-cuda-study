# 002 - Vector Addition

- Track: `Foundations`
- Difficulty: `Beginner`
- Status: `Reference-friendly`
- GitHub batch: `001-020`

## Goal

Build and study a working CUDA implementation of **Vector Addition**.

This is the canonical CUDA "hello world" example. It demonstrates:

- basic kernel functions
- memory allocation on the GPU
- data transfer between host and device
- CPU/GPU result checking

## PMPP Ideas To Focus On

- 1D indexing
- coalesced access
- CPU vs GPU validation

## What You Should Learn Here

- How one thread maps to one array element
- How `cudaMalloc`, `cudaMemcpy`, and a kernel launch fit together
- Why this is the simplest complete CUDA workflow worth mastering first

## Study Prompts

- Identify where host memory ends and device memory begins in `main.cu`.
- Explain why the bounds check is still needed in such a simple kernel.
- Rewrite the kernel as a grid-stride loop and compare the structure.

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Validation

- The program prints `PASS` when GPU output matches the CPU reference or expected pattern.
- Start with the built-in small inputs before scaling up.

## What To Modify Next

- Switch to a grid-stride loop.
- Benchmark larger arrays.
- Add CUDA event timing.
