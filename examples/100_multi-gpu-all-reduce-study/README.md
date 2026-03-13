# 100 - Multi GPU All Reduce Study

- Track: `Graph and ML`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Study an all-reduce pattern by simulating two device shards, a reduce step, and a broadcast step in one CUDA program.

## PMPP Ideas To Focus On

- collective communication structure
- partitioned tensor updates
- honesty about single-process study scaffolds versus real multi-GPU execution

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This example simulates the dataflow of all-reduce on one device so the algorithm is inspectable without requiring multi-GPU hardware.
- It is best read as a conceptual stepping stone toward NCCL-based implementations.
- A next step is replacing the virtual shards with real device-local buffers on multiple GPUs.
