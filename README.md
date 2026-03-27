# PMPP CUDA Study

Structured CUDA study repository with **150 examples**: **100 core PMPP-style examples** plus **50 advanced studies**. The repo is organized as a disciplined progression rather than a loose kernel dump: early examples establish correctness, indexing, memory movement, and synchronization; later examples widen into sparse kernels, imaging, simulation, graph workloads, and ML-oriented operators.

Project site: [manishklach.github.io/pmpp-cuda-study](https://manishklach.github.io/pmpp-cuda-study/)

## Repo Identity

This repo exists to make CUDA study concrete. Each example is meant to be small enough to read in one sitting, but complete enough to compile, run, validate against a CPU reference where appropriate, and compare with a nearby baseline or optimized variant when that comparison teaches something real.

The repository now has two tracks:

| Track | Range | Purpose |
|---|---|---|
| Core PMPP-style study track | `001-100` | Foundational CUDA progression: kernels, memory behavior, reduction, scan, histograms, dense linear algebra, image processing, simulation, graph, and ML basics |
| Advanced Studies | `101-150` | Focused studies on warp primitives, atomics, tiling, sparse / irregular workloads, simulation, and practical ML kernels |

## Progress / Status

| Status | Count | Notes |
|---|---:|---|
| Implemented | 110 | Runnable CUDA examples with validation and README guidance |
| Scaffolded | 40 | Structured placeholders with curriculum notes and future implementation guidance |
| Planned | 0 | The current repo now contains the full `001-150` sequence |

## Implemented So Far

- Core track:
  `001-056` and `061-100` are runnable; `057-060` remain scaffolded for factorization-heavy topics.
- Advanced studies:
  `101_segmented-reduction`, `102_segmented-scan`, `105_warp-aggregated-atomics`, `111_warp-shuffle-reduction`, `112_warp-shuffle-scan`, `116_bank-conflict-study`, `117_coalescing-study`, `120_stencil-with-halo-tiling`, `131_sobel-filter-optimized`, `137_heat-diffusion-tiled-2d`, `141_layernorm-forward`, `142_softmax-stable`, `149_monte-carlo-gbm-option-pricing`, and `150_mini-inference-pipeline` are fully implemented.

## Best Examples To Start With

- `002_vector-addition`: the cleanest host-device workflow baseline
- `020_matrix-transpose-with-shared-memory`: early shared-memory reuse with clear 2D mapping
- `023_sum-reduction`: the core PMPP reduction pattern
- `043_tiled-matrix-multiply`: classic reuse-driven optimization
- `111_warp-shuffle-reduction`: the first strong warp-primitive study
- `141_layernorm-forward`: a practical ML operator built from familiar reduction ideas

## Best Examples For Optimization Study

- `020_matrix-transpose-with-shared-memory`
- `026_prefix-sum-naive-scan`
- `027_prefix-sum-work-efficient-scan`
- `028_histogram-global-atomics`
- `029_histogram-shared-memory`
- `042_naive-matrix-multiply`
- `043_tiled-matrix-multiply`
- `116_bank-conflict-study`
- `117_coalescing-study`
- `120_stencil-with-halo-tiling`

## Best Examples For Interview Prep / Learning CUDA Patterns

- `002_vector-addition`: launch configuration, indexing, and validation
- `023_sum-reduction`: shared memory, divergence, and synchronization
- `027_prefix-sum-work-efficient-scan`: up-sweep / down-sweep reasoning
- `029_histogram-shared-memory`: privatization and contention tradeoffs
- `030_stream-compaction`: scan-driven filtering
- `111_warp-shuffle-reduction`: warp-synchronous programming
- `142_softmax-stable`: reduction logic inside a practical ML primitive

## How To Study This Repo

1. Start with correctness. Build the example, run `--check`, and make sure you can explain why the CPU reference is trustworthy.
2. Read the simpler version first when there is one. Good pairs include `026` before `027`, `028` before `029`, `042` before `043`, and `101` before `111`.
3. Study mapping and memory behavior. Ask which thread owns which output, which values are reused, where atomics appear, and where synchronization is mandatory.
4. Move to the advanced studies after the core pattern feels stable. The advanced track assumes you already recognize reduction trees, scan structure, tiling, and common memory bottlenecks.
5. Benchmark last. Use `--bench` to confirm what the code suggests rather than replacing reasoning with timing.

## Advanced Studies (101-150)

| Group | Range | Implemented | Scaffolded | Focus |
|---|---|---:|---:|---|
| Warp / atomics / scan | `101-110` | 3 | 7 | Segmented operations, warp-aggregated atomics, fused prefix-sum style kernels |
| Memory / tiling / optimization | `111-120` | 5 | 5 | Warp shuffle, tiling, bank conflicts, coalescing, halo staging |
| Sparse / graph / irregular | `121-130` | 0 | 10 | Sparse matrix formats, graph frontiers, irregular gather / scatter |
| Imaging / simulation | `131-140` | 2 | 8 | Filters, FFT / sort studies, blocked solvers, simulation kernels |
| ML / practical kernels | `141-150` | 4 | 6 | Normalization, softmax, attention-adjacent kernels, pricing, mini pipelines |

## Curriculum Summary

| Module | Example Range | Implemented | Scaffolded | Notes |
|---|---|---:|---:|---|
| Foundations | `001-020` | 20 | 0 | Complete beginner path |
| Parallel patterns | `021-040` | 20 | 0 | Reduction, scan, histogram, compaction, sorting, and search |
| Linear algebra | `041-060` | 16 | 4 | Dense and sparse kernels are implemented; heavy factorizations remain scaffolded |
| Image and signal | `061-075` | 15 | 0 | Runnable image and signal-processing progression |
| Simulation | `076-090` | 15 | 0 | Runnable simulation and rendering progression |
| Graph and ML | `091-100` | 10 | 0 | Runnable graph / ML progression |
| Advanced studies | `101-150` | 14 | 36 | Structured second track for optimization, irregular workloads, and practical kernels |

## Build Instructions

Direct `nvcc` build from an example folder:

```powershell
cd examples\111_warp-shuffle-reduction
nvcc -std=c++17 -O2 -I..\..\include main.cu -o example.exe
.\example.exe --check --size 65536
```

Top-level CMake build:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

Assumptions:

- CUDA Toolkit with `nvcc` is installed and on `PATH`
- examples target C++17
- benchmark output is illustrative and intended for local comparison, not publication-grade performance claims

## Benchmarking Philosophy

Correctness first, then performance. The examples use deterministic inputs, CPU reference checks where appropriate, and a lightweight warmup plus timed-iteration pattern so nearby kernels can be compared without hiding the algorithm. These are educational microbenchmarks. They are useful for studying memory behavior, synchronization costs, and baseline-versus-improved structure, but they are not substitutes for production benchmarking on controlled hardware.

## Supporting Docs

- [docs/example-conventions.md](docs/example-conventions.md)
- [docs/maturity-model.md](docs/maturity-model.md)
- [docs/status.md](docs/status.md)
- [scripts/validate_repo.py](scripts/validate_repo.py)
- [scripts/build_examples.py](scripts/build_examples.py)
