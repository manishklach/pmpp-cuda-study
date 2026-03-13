# Status Tracking

Use this file as the source of truth for repo implementation status.

## Status Legend

- `✅ fully mature` — validated, benchmarkable, documentation-rich example
- `🧪 verified` — validated example with clear PASS / FAIL behavior
- `⚙️ compiles` — builds, but validation maturity is still limited
- `🟡 scaffolded` — starter code and docs exist, but the example is not yet trusted
- `📝 notes only` — conceptual planning exists, but code is not added yet
- `🚧 in progress` — currently being implemented or revised

See [docs/maturity-model.md](maturity-model.md) for the repo-wide maturity ladder.

## Current High-Level Status

| Module | Example Range | Count | Implemented | Scaffolded | Notes Only | Comments |
|---|---:|---:|---:|---:|---:|---|
| Foundations | 001-012 | 12 | 12 | 0 | 0 | Strong beginner path |
| Memory and execution basics | 013-022 | 10 | 0 | 10 | 0 | Good next expansion target |
| Reductions / scans / compaction | 023-036 | 14 | 7 | 7 | 0 | Extend scans and segmented work |
| Histograms / sorting / irregular primitives | 037-048 | 12 | 5 | 7 | 0 | Sorting and histogram ladders are high value |
| Matrix / stencil / convolution | 049-070 | 22 | 18 | 4 | 0 | Factorization-heavy examples can remain scaffold-like |
| Sparse / graph workloads | 071-086 | 16 | 0 | 16 | 0 | Planned expansion module |
| ML / tensor kernels | 087-104 | 18 | 0 | 18 | 0 | Planned expansion module |
| Simulation / rendering / scientific computing | 105-120 | 16 | 0 | 16 | 0 | Planned expansion module |
| Performance engineering | 121-132 | 12 | 0 | 12 | 0 | Planned expansion module |
| Capstones / optimization ladders | 133-140 | 8 | 0 | 8 | 0 | Planned expansion module |

## Per-Example Tracking Template

```markdown
| # | Slug | Title | Module | Difficulty | Status | Validation | Notes |
|---:|---|---|---|---|---|---|---|
| 002 | vector-add | Vector Addition | Foundations | beginner | ✅ implemented | CPU reference | Canonical CUDA hello world |
| 023 | sum-reduction | Sum Reduction | Reductions / scans / compaction | intermediate | ✅ implemented | CPU reference | Shared-memory reduction baseline |
| 043 | tiled-matrix-multiply | Tiled Matrix Multiply | Matrix / stencil / convolution | intermediate | ✅ implemented | CPU reference | Core tiling example |
| 073 | spmv-csr | SpMV CSR | Sparse / graph workloads | advanced | 🟡 scaffolded | pending | High-value next implementation |
| 091 | softmax-rowwise | Softmax Rowwise | ML / tensor kernels | advanced | 📝 notes only | none | Good ML bridge kernel |
| 134 | reduction-optimization-ladder | Reduction Optimization Ladder | Capstones / optimization ladders | advanced | 🚧 in progress | benchmark + correctness | Compare multiple implementations |
```

## Suggested Workflow

1. Update module counts when a batch lands.
2. Update the per-example table for any example that changes status.
3. Keep status honest: scaffolding is valuable, but it should not be labeled mature.
4. Add links to reports or benchmark summaries in the `Notes` column where useful.

## Current Trusted Core

The strongest current maturity path is:

- `002_vector-addition`
- `007_saxpy`
- `023_sum-reduction`
- `042_naive-matrix-multiply`
- `043_tiled-matrix-multiply`
- `049_gaussian-blur`
- `080_n-body-tiled`
