# 077 - Monte Carlo Option Pricing

- Track: `Simulation`
- Difficulty: `Advanced`
- Status: `Reference-friendly`
- GitHub batch: `061-080`

## Goal

Estimate the price of a European call option with one Monte Carlo path per thread and compare it against a CPU path simulation using the same pseudo-random generator logic.

## PMPP Ideas To Focus On

- parallel Monte Carlo paths
- financial payoff aggregation
- using Box-Muller normals without external libraries

## Build

```powershell
nvcc -std=c++17 -O2 main.cu -o example.exe
```

## Run

```powershell
.\example.exe
```

## Study Notes

- This stays self-contained by using a simple LCG and Box-Muller transform instead of CURAND.
- For teaching, the option parameters are modest and the validation tolerance is practical rather than strict.
- A next step is variance reduction or multi-step path simulation.
