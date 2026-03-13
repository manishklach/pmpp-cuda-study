# 095 - Union Find

- Track: `Graph and ML`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Build a small disjoint-set structure on the GPU with repeated union passes and path compression.

## PMPP Ideas To Focus On

- pointer-jumping style updates
- repeated convergence passes
- set merging for graph connectivity work

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This is a compact study version rather than a lock-free production union-find.
- The repeated compression pass makes the forest structure easier to reason about.
- A next step is combining this with connected-components labeling or Kruskal-style workflows.
