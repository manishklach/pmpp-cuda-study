# 091 - Parallel BFS

- Track: `Graph and ML`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Traverse a small graph in breadth-first order using level-synchronous frontier expansion on the GPU.

## PMPP Ideas To Focus On

- frontier-based irregular parallelism
- CSR graph storage
- repeated kernel launches across levels

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is a compact version of the classic PMPP BFS structure.
- The host controls the outer level loop while the GPU expands one frontier at a time.
- A next step is experimenting with push-pull traversal strategies.
