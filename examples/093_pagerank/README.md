# 093 - PageRank

- Track: `Graph and ML`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `081-100`

## Goal

Run a few PageRank iterations on a tiny directed graph using one contribution pass and one normalization pass per iteration.

## PMPP Ideas To Focus On

- iterative graph analytics
- atomic accumulation from outgoing edges
- repeated kernel launches with damping

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This keeps the graph tiny so you can verify the rank updates by hand.
- The implementation uses edge-parallel contribution accumulation for clarity.
- A next step is handling dangling nodes and larger sparse graphs more carefully.
