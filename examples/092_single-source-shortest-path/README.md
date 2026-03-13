# 092 - Single Source Shortest Path

- Track: `Graph and ML`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Solve a small weighted graph with Bellman-Ford style edge relaxations on the GPU.

## PMPP Ideas To Focus On

- parallel edge processing
- repeated relaxation rounds
- graph algorithm correctness over irregular memory access

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- Bellman-Ford is a good teaching baseline before more advanced SSSP variants.
- Each relaxation pass is embarrassingly parallel over edges.
- A next step is adding frontier filtering or delta-stepping ideas.
