# 098 - Neural Network Forward Pass

- Track: `Graph and ML`
- Difficulty: `Advanced`
- Status: `🧪 verified`
- Maturity: `Level 4 - benchmarkable`

## Goal

Run a small two-layer MLP forward pass on the GPU and validate both hidden activations and final outputs against a CPU reference.

## Why This Example Matters

This gives the repo a more trustworthy ML anchor. It links the earlier dense-kernel material to neural network inference without requiring large frameworks.

## CUDA Concepts Taught

- dense layer forward pass
- activation functions
- multi-kernel inference pipelines
- CPU reference validation for ML kernels

## Build

```powershell
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
```

## Run

```powershell
.\example.exe --check --size 32
```

```powershell
.\example.exe --bench --size 128 --warmup 5 --iters 20
```

## Expected Output

- Prints `PASS` when hidden and output activations match the CPU reference within tolerance.

## Next Optimization Steps

- batch multiple inputs
- compare ReLU and non-ReLU variants
- add softmax or layernorm as the next mature ML examples
