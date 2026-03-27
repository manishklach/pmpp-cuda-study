# Status Tracking

This file is the compact status view for the **150-example** PMPP CUDA Study repository.

## Status Legend

- `Implemented`: runnable example with code, README guidance, and at least a direct correctness path
- `Scaffolded`: structured placeholder with topic framing, expected kernel shape, and future implementation notes
- `Planned`: reserved for future additions outside the current `001-150` sequence

## Current Snapshot

| Bucket | Count | Notes |
|---|---:|---|
| Implemented | 110 | Core study track plus the first advanced batch |
| Scaffolded | 40 | `057-060` plus advanced studies not yet implemented |
| Planned | 0 | No empty slots in the current `001-150` sequence |

## Track Summary

| Track | Range | Implemented | Scaffolded | Notes |
|---|---|---:|---:|---|
| Core PMPP-style study track | `001-100` | 96 | 4 | Main progression from beginner kernels to graph / ML basics |
| Advanced studies | `101-150` | 14 | 36 | Structured second track for deeper optimization and irregular workloads |

## Core Modules

| Module | Example Range | Implemented | Scaffolded | Notes |
|---|---|---:|---:|---|
| Foundations | `001-020` | 20 | 0 | Strong beginner path |
| Parallel patterns | `021-040` | 20 | 0 | Includes polished reduction, scan, histogram, and compaction examples |
| Linear algebra | `041-060` | 16 | 4 | Dense kernels are implemented; factorization-heavy topics remain scaffolded |
| Image and signal | `061-075` | 15 | 0 | Runnable image and signal-processing examples |
| Simulation | `076-090` | 15 | 0 | Runnable simulation and rendering progression |
| Graph and ML | `091-100` | 10 | 0 | Runnable graph / ML progression |

## Advanced Studies

| Group | Example Range | Implemented | Scaffolded | Notes |
|---|---|---:|---:|---|
| Warp / atomics / scan | `101-110` | 3 | 7 | Segmented reduction / scan, warp aggregation, radix-style filtering studies |
| Memory / tiling / optimization | `111-120` | 5 | 5 | Warp shuffle, bank-conflict, coalescing, transpose, and halo tiling studies |
| Sparse / graph / irregular | `121-130` | 0 | 10 | Sparse formats, graph traversal, hashing, and irregular access studies |
| Imaging / simulation | `131-140` | 2 | 8 | Filter optimization, blocked solvers, simulation kernels, and visual studies |
| ML / practical kernels | `141-150` | 4 | 6 | Normalization, softmax, attention-adjacent, finance, and small inference kernels |

## Implemented Advanced Examples

- `101_segmented-reduction`
- `102_segmented-scan`
- `105_warp-aggregated-atomics`
- `111_warp-shuffle-reduction`
- `112_warp-shuffle-scan`
- `116_bank-conflict-study`
- `117_coalescing-study`
- `120_stencil-with-halo-tiling`
- `131_sobel-filter-optimized`
- `137_heat-diffusion-tiled-2d`
- `141_layernorm-forward`
- `142_softmax-stable`
- `149_monte-carlo-gbm-option-pricing`
- `150_mini-inference-pipeline`

## Maintenance Notes

1. Keep the root [README](../README.md), this file, and the GitHub Pages site aligned on the `150`-example story and the `001-100` versus `101-150` split.
2. If an advanced study is promoted from scaffolded to implemented, update both the group counts and the top-level snapshot.
3. Use `scripts/validate_repo.py` for structure checks and `scripts/build_examples.py` for `nvcc` build validation when CUDA is available.
