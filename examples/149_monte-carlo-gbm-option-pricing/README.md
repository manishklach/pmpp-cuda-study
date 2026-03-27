# 149 Monte Carlo GBM Option Pricing

## Overview

This example prices a European call option with a simple Monte Carlo simulation under geometric Brownian motion. Each thread simulates one path, computes the payoff, and a second reduction kernel accumulates the total payoff before discounting.

## What this example teaches

- how embarrassingly parallel Monte Carlo work maps naturally to CUDA threads
- why deterministic random-number generation matters for validation
- how a practical finance kernel still depends on a clean reduction step

## CUDA concepts involved

- one-thread-per-path mapping
- device-side pseudo-random generation
- payoff accumulation and block reduction
- host-side final aggregation

## Kernel mapping

- simulation kernel: one thread per path
- reduction kernel: one block reduces a contiguous chunk of path payoffs
- the host combines block sums into the final discounted option price

## Memory behavior

The simulation kernel writes one payoff per path to global memory. The reduction kernel then reads those payoffs coalesced and stages them in shared memory for summation. There is no inter-thread communication in the path simulation itself.

## Correctness approach

A CPU reference uses the same deterministic linear-congruential generator and Box-Muller transform. The final GPU option price is compared against the CPU price with a scalar tolerance.

## Build and run

```powershell
cd examples\149_monte-carlo-gbm-option-pricing
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 16384 --seed 149
.\example.exe --bench --size 1048576 --warmup 5 --iters 20 --seed 149
```

## Expected output

```text
Example: 149_monte-carlo-gbm-option-pricing
Mode: check
Validation: PASS
```

## Common mistakes

- using nondeterministic RNG state and then trying to debug correctness
- forgetting to discount the expected payoff back to present value
- comparing path-by-path noise instead of comparing the final expected price

## Possible optimizations / next step

A good next step is to generate multiple paths per thread and use quasi-random or low-discrepancy sequences, then explore variance reduction techniques such as antithetic sampling.
