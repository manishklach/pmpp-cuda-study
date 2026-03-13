# Example Maturity Model

This repo uses a simple maturity ladder so example quality is explicit.

## Levels

| Level | Label | Meaning |
|---|---|---|
| 0 | Placeholder | Topic exists in planning or docs only. |
| 1 | Scaffolded | Folder, docs, and starter code exist, but the example is not yet trusted. |
| 2 | Compiles | Example builds successfully in the intended CUDA environment. |
| 3 | Verified | Example validates GPU output against a CPU reference or deterministic expected behavior. |
| 4 | Benchmarkable | Example supports standard benchmark mode with timing controls. |
| 5 | Variant-Aware | Example includes a clear baseline-to-improved comparison path. |
| 6 | Polished Teaching Example | Example is documentation-rich, validated, benchmarkable, and suitable as a flagship study artifact. |

## Repo Status Labels

- `✅ fully mature` maps to Levels `4-6`
- `🧪 verified` maps to Level `3`
- `⚙️ compiles` maps to Level `2`
- `🟡 scaffolded` maps to Level `1`
- `📝 notes only` maps to Level `0`
- `🚧 in progress` means the example is being actively revised

## Standard Expectations For Mature Examples

A mature example should:

- build with a standard `nvcc` command
- expose `--check` and optionally `--bench`
- validate GPU output
- print clear `PASS` or `FAIL`
- return nonzero on failure
- include `README.md` and `meta.yaml`
- document likely bottlenecks and next optimization steps
