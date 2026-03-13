# 099 - MLP Backpropagation

- Track: `Graph and ML`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Compute output-layer gradients for a tiny MLP using mean-squared error and a fixed hidden activation vector.

## PMPP Ideas To Focus On

- parallel gradient computation across weights
- separating forward activations from backward signals
- building intuition for training kernels before optimizer logic

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This keeps backprop focused on one layer so the gradient formulas stay visible.
- The output includes bias and weight gradients for a deterministic target.
- A next step is extending the gradient chain through the hidden layer.
