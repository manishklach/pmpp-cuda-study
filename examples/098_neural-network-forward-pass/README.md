# 098 - Neural Network Forward Pass

- Track: `Graph and ML`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Compute a small two-layer MLP forward pass with ReLU activation on the GPU.

## PMPP Ideas To Focus On

- dense linear algebra as batched neuron evaluation
- activation functions as elementwise kernels
- mapping model layers into simple CUDA stages

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is the simplest end-to-end neural network example in the repo.
- The implementation favors readability over GEMM-level optimization.
- A next step is batching multiple inputs or replacing the dense loops with cuBLAS.
