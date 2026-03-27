# 150 Mini Inference Pipeline

## Overview

This example chains two small linear layers with a ReLU in between and a final row-wise softmax. It is not meant to be a production inference engine. It is meant to show how several familiar CUDA patterns compose into a tiny end-to-end pipeline.

## What this example teaches

- how separate dense kernels can be composed into a simple forward pass
- why correctness checks matter even when each individual stage looks straightforward
- how linear algebra, activation, and normalization patterns show up together in inference-style code

## CUDA concepts involved

- one-thread-per-output dense layers
- row-wise softmax reduction
- kernel sequencing and intermediate buffers
- host-side orchestration of a small inference graph

## Kernel mapping

- linear layers: 2D grid, one thread per output activation
- ReLU: fused into the first linear layer for clarity
- softmax: one block per batch row of logits

## Memory behavior

The pipeline uses explicit intermediate buffers so each stage remains easy to read. That means more global-memory traffic than a fused production kernel, but it makes the handoff between layers, activation, and softmax visible.

## Correctness approach

A CPU reference computes the same two-layer forward pass and stable softmax. The final probability matrix is compared against the GPU result with a floating-point tolerance.

## Build and run

```powershell
cd examples\150_mini-inference-pipeline
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check
.\example.exe --bench --warmup 5 --iters 20
```

## Expected output

```text
Example: 150_mini-inference-pipeline
Mode: check
Validation: PASS
```

## Common mistakes

- mixing up matrix shapes between the two dense layers
- forgetting that softmax is row-wise over output classes
- overwriting intermediate buffers too early when sequencing kernels

## Possible optimizations / next step

The obvious next step is fusion: tiled GEMM for the dense layers, warp-specialized softmax, and eventually a single kernel that reduces intermediate memory traffic for small-batch inference.
