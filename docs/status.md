# Status Tracking

This file is the compact status view for the 100-example PMPP CUDA study sequence.

## Status Legend

- `Implemented`: runnable example with code, README notes, and at least a direct correctness path
- `Template`: intentionally scaffolded topic where a lightweight study sketch is clearer than pretending to be a full implementation
- `Planned`: reserved for future additions outside the current 100-example set

## Current Snapshot

| Bucket | Count | Notes |
|---|---:|---|
| Implemented | 96 | Includes the full `001-056` range and `061-100` |
| Template | 4 | `057_lu-factorization-sketch`, `058_cholesky-factorization`, `059_qr-factorization-sketch`, `060_fft-based-convolution` |
| Planned | 0 | No empty slots in the current 100-example sequence |

## Module Summary

| Module | Example Range | Implemented | Template | Notes |
|---|---|---:|---:|---|
| Foundations | 001-020 | 20 | 0 | Strong beginner path |
| Parallel patterns | 021-040 | 20 | 0 | Includes polished reduction, scan, histogram, and compaction examples |
| Linear algebra | 041-060 | 16 | 4 | Dense kernels are implemented; factorization-heavy topics remain scaffolded |
| Image and signal | 061-075 | 15 | 0 | Runnable image and signal-processing examples |
| Simulation | 076-090 | 15 | 0 | Runnable simulation and rendering progression |
| Graph and ML | 091-100 | 10 | 0 | Runnable graph / ML progression |

## Trusted Study Core

Recommended examples when you want the strongest current path through the repo:

- `002_vector-addition`
- `020_matrix-transpose-with-shared-memory`
- `021_dot-product`
- `023_sum-reduction`
- `026_prefix-sum-naive-scan`
- `027_prefix-sum-work-efficient-scan`
- `028_histogram-global-atomics`
- `029_histogram-shared-memory`
- `030_stream-compaction`
- `041_matrix-vector-multiply`
- `042_naive-matrix-multiply`
- `043_tiled-matrix-multiply`

## Maintenance Notes

1. Keep the root [README](../README.md) and this file aligned on counts and template examples.
2. If an example is promoted from template to implemented, update both the module summary and the root example table.
3. Use `scripts/validate_repo.py` for structure checks and `scripts/build_examples.py` for `nvcc` build validation when CUDA is available.
