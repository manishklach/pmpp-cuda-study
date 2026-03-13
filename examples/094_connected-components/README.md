# 094 - Connected Components

- Track: `Graph and ML`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Find connected components on a small undirected graph using iterative label propagation.

## PMPP Ideas To Focus On

- iterative convergence on irregular graphs
- edge-parallel label updates
- comparing graph kernels against a simple CPU baseline

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- Label propagation is simple to follow and great for small study graphs.
- The repeated passes make the convergence behavior visible.
- A next step is experimenting with union-find or hooking this to BFS-style frontiers.
