from pathlib import Path
from textwrap import dedent
import re

ROOT = Path(r"C:\Users\ManishKL\Documents\Playground\pmpp-cuda-study")
EXAMPLES = ROOT / "examples"

FIRST20 = {
    1: (["kernel launch syntax", "thread/block coordinates", "device synchronization"], ["Change the launch geometry.", "Record more thread metadata on the device."]),
    2: (["1D indexing", "coalesced access", "CPU vs GPU validation"], ["Switch to a grid-stride loop.", "Benchmark larger arrays."]),
    3: (["reusing elementwise patterns", "bounds checks", "correctness tests"], ["Use random negative values.", "Compare with vector addition."]),
    4: (["kernel parameters", "simple arithmetic throughput", "reference checks"], ["Read the scalar from argv.", "Try doubles if your GPU supports them."]),
    5: (["embarrassingly parallel transforms", "numeric checks", "launch sizing"], ["Replace square with cube.", "Add timing after validation."]),
    6: (["math intrinsics", "branchless logic", "signed inputs"], ["Compare against a branchy version.", "Swap to integer input."]),
    7: (["BLAS-style vector ops", "fused arithmetic", "bandwidth-bound kernels"], ["Generalize to z = a*x + b*y.", "Compare different block sizes."]),
    8: (["memory movement", "minimal kernels", "launch overhead"], ["Compare with cudaMemcpy when CUDA is available.", "Use float4 or uchar4 data."]),
    9: (["index remapping", "read/write patterns", "easy-to-inspect test data"], ["Try in-place reversal.", "Test odd and even lengths."]),
    10: (["conditional kernels", "min/max patterns", "range normalization"], ["Parameterize the clamp interval.", "Count clipped elements."]),
    11: (["predicate kernels", "binary outputs", "simple segmentation"], ["Emit 0/255 instead of 0/1.", "Add lower and upper thresholds."]),
    12: (["pixel structs", "image indexing", "weighted color transforms"], ["Try alternative luminance weights.", "Preserve alpha in a uchar4 variant."]),
    13: (["byte-wise image transforms", "RGB buffer layout", "sample inspection"], ["Invert a single channel only.", "Operate on grayscale data."]),
    14: (["saturating arithmetic", "parameterized kernels", "image transforms"], ["Try negative deltas.", "Process a larger image."]),
    15: (["affine transforms", "byte clamping", "visual intuition"], ["Vary the contrast factor.", "Change the midpoint."]),
    16: (["neighbor access", "boundary conditions", "stencil decomposition"], ["Expand to a 5-point stencil.", "Add a shared-memory version later."]),
    17: (["2D indexing", "halo boundaries", "grid launch geometry"], ["Swap box average for Laplacian.", "Add shared-memory tiling later."]),
    18: (["2D launches", "flattened storage", "matrix validation"], ["Use rectangular matrices.", "Compare 1D vs 2D launches."]),
    19: (["row-major indexing", "strided writes", "baseline transpose"], ["Try rectangular inputs.", "Time it against the tiled version later."]),
    20: (["shared memory tiles", "synchronization", "avoiding bank conflicts"], ["Change tile size to 16 or 32.", "Compare against the naive transpose."]),
}


def write(path: Path, text: str) -> None:
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")


def title_from_readme(example_dir: Path) -> str:
    first = example_dir.joinpath("README.md").read_text(encoding="utf-8").splitlines()[0]
    return first.split(" - ", 1)[1].strip()


def track(index: int) -> str:
    if index <= 20:
        return "Foundations"
    if index <= 40:
        return "Parallel Patterns"
    if index <= 60:
        return "Linear Algebra"
    if index <= 75:
        return "Image and Signal"
    if index <= 90:
        return "Simulation"
    return "Graph and ML"


def difficulty(index: int) -> str:
    if index <= 15:
        return "Beginner"
    if index <= 50:
        return "Intermediate"
    return "Advanced"


def status(index: int) -> str:
    return "Reference-friendly" if index <= 20 else "Guided template"


def batch(index: int) -> str:
    start = ((index - 1) // 20) * 20 + 1
    return f"{start:03d}-{start + 19:03d}"


def concepts(index: int) -> list[str]:
    return {
        "Foundations": ["thread indexing", "bounds checks", "host-device memory flow"],
        "Parallel Patterns": ["work decomposition", "shared memory or atomics", "validation before tuning"],
        "Linear Algebra": ["data layout", "memory reuse", "correctness against a CPU reference"],
        "Image and Signal": ["2D or chunk indexing", "boundary handling", "pipeline composition"],
        "Simulation": ["state updates", "time stepping or sampling", "numerical checks"],
        "Graph and ML": ["irregular parallelism", "iteration strategy", "scalability planning"],
    }[track(index)]


def readme_first20(index: int, title: str) -> str:
    focus, modify = FIRST20[index]
    lines = [
        f"# {index:03d} - {title}",
        "",
        "- Track: `Foundations`",
        f"- Difficulty: `{difficulty(index)}`",
        "- Status: `Reference-friendly`",
        "- GitHub batch: `001-020`",
        "",
        "## Goal",
        "",
        f"Build and study a working CUDA implementation of **{title}**.",
        "",
        "## PMPP Ideas To Focus On",
        "",
    ]
    lines.extend(f"- {x}" for x in focus)
    lines.extend([
        "",
        "## Build",
        "",
        "```powershell",
        "nvcc -std=c++17 -O2 main.cu -o example.exe",
        "```",
        "",
        "## Run",
        "",
        "```powershell",
        ".\\example.exe",
        "```",
        "",
        "## Validation",
        "",
        "- The program prints `PASS` when GPU output matches the CPU reference or expected pattern.",
        "- Start with the built-in small inputs before scaling up.",
        "",
        "## What To Modify Next",
        "",
    ])
    lines.extend(f"- {x}" for x in modify)
    return "\n".join(lines)


def readme_generic(index: int, title: str) -> str:
    lines = [
        f"# {index:03d} - {title}",
        "",
        f"- Track: `{track(index)}`",
        f"- Difficulty: `{difficulty(index)}`",
        f"- Status: `{status(index)}`",
        f"- GitHub batch: `{batch(index)}`",
        "",
        "## Goal",
        "",
        f"Study **{title}** in CUDA using a PMPP-style decomposition. Start small, validate correctness, then tune.",
        "",
        "## PMPP Ideas To Focus On",
        "",
    ]
    lines.extend(f"- {x}" for x in concepts(index))
    lines.extend([
        "",
        "## Build",
        "",
        "```powershell",
        "nvcc -std=c++17 -O2 main.cu -o example.exe",
        "```",
        "",
        "## Run",
        "",
        "```powershell",
        ".\\example.exe",
        "```",
        "",
        "## Study Checklist",
        "",
        "- Describe the parallel unit of work.",
        "- Explain the launch configuration.",
        "- Compare GPU output against a CPU reference.",
        "- Note one correctness risk and one performance risk.",
        "- Write one extension you want to try next.",
    ])
    return "\n".join(lines)


def shared_header(index: int, title: str) -> str:
    return dedent(f"""
    // Example {index:03d}: {title}
    // Track: {track(index)}
    // Difficulty: {difficulty(index)}
    // Status: {status(index)}

    #include <cuda_runtime.h>
    #include <cmath>
    #include <cstdlib>
    #include <iostream>
    #include <vector>

    #define CHECK_CUDA(call) \\
      do {{ \\
        cudaError_t status__ = (call); \\
        if (status__ != cudaSuccess) {{ \\
          std::cerr << "CUDA error: " << cudaGetErrorString(status__) \\
                    << " at " << __FILE__ << ":" << __LINE__ << std::endl; \\
          std::exit(EXIT_FAILURE); \\
        }} \\
      }} while (0)
    """)


def vector_binary(index: int, title: str, gpu_expr: str, cpu_expr: str) -> str:
    return shared_header(index, title) + dedent(f"""
    __global__ void kernel(const float* a, const float* b, float* out, int n) {{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n) {{
        out[idx] = {gpu_expr};
      }}
    }}

    int main() {{
      const int n = 1 << 12;
      const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
      std::vector<float> a(n), b(n), gpu_out(n, 0.0f), cpu_out(n, 0.0f);
      for (int i = 0; i < n; ++i) {{
        float lhs = static_cast<float>((i % 29) - 14) * 0.25f;
        float rhs = static_cast<float>((i % 13) - 6) * 0.5f;
        a[i] = lhs;
        b[i] = rhs;
        cpu_out[i] = {cpu_expr};
      }}

      float* d_a = nullptr;
      float* d_b = nullptr;
      float* d_out = nullptr;
      CHECK_CUDA(cudaMalloc(&d_a, bytes));
      CHECK_CUDA(cudaMalloc(&d_b, bytes));
      CHECK_CUDA(cudaMalloc(&d_out, bytes));
      CHECK_CUDA(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

      const int threads = 256;
      const int blocks = (n + threads - 1) / threads;
      kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaMemcpy(gpu_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

      int mismatches = 0;
      for (int i = 0; i < n; ++i) {{
        if (std::fabs(gpu_out[i] - cpu_out[i]) > 1.0e-5f) {{
          ++mismatches;
        }}
      }}

      std::cout << "Validation: " << (mismatches == 0 ? "PASS" : "FAIL") << std::endl;
      CHECK_CUDA(cudaFree(d_a));
      CHECK_CUDA(cudaFree(d_b));
      CHECK_CUDA(cudaFree(d_out));
      return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }}
    """)


def vector_unary(index: int, title: str, gpu_expr: str, cpu_expr: str) -> str:
    return shared_header(index, title) + dedent(f"""
    __global__ void kernel(const float* input, float* out, int n) {{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n) {{
        out[idx] = {gpu_expr};
      }}
    }}

    int main() {{
      const int n = 1 << 12;
      const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
      std::vector<float> input(n), gpu_out(n, 0.0f), cpu_out(n, 0.0f);
      for (int i = 0; i < n; ++i) {{
        float value = static_cast<float>((i % 23) - 11) * 0.25f;
        input[i] = value;
        cpu_out[i] = {cpu_expr};
      }}

      float* d_input = nullptr;
      float* d_out = nullptr;
      CHECK_CUDA(cudaMalloc(&d_input, bytes));
      CHECK_CUDA(cudaMalloc(&d_out, bytes));
      CHECK_CUDA(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

      const int threads = 256;
      const int blocks = (n + threads - 1) / threads;
      kernel<<<blocks, threads>>>(d_input, d_out, n);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaMemcpy(gpu_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

      int mismatches = 0;
      for (int i = 0; i < n; ++i) {{
        if (std::fabs(gpu_out[i] - cpu_out[i]) > 1.0e-5f) {{
          ++mismatches;
        }}
      }}

      std::cout << "Validation: " << (mismatches == 0 ? "PASS" : "FAIL") << std::endl;
      CHECK_CUDA(cudaFree(d_input));
      CHECK_CUDA(cudaFree(d_out));
      return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }}
    """)


def generic_code(index: int, title: str) -> str:
    notes = "\n".join(f"// - Study focus: {item}" for item in concepts(index))
    return shared_header(index, title) + dedent(f"""
    {notes}

    __global__ void study_kernel(const float* a, const float* b, float* out, int n) {{
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n) {{
        out[idx] = a[idx] + b[idx];
      }}
    }}

    static void fill_input(std::vector<float>& values, float scale) {{
      for (int i = 0; i < static_cast<int>(values.size()); ++i) {{
        values[i] = scale * static_cast<float>((i % 17) - 8);
      }}
    }}

    static void cpu_reference(const std::vector<float>& a,
                              const std::vector<float>& b,
                              std::vector<float>& out) {{
      for (int i = 0; i < static_cast<int>(out.size()); ++i) {{
        out[i] = a[i] + b[i];
      }}
    }}

    int main() {{
      std::cout << "Running {index:03d}" << std::endl;

      const int n = 1 << 12;
      const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
      std::vector<float> host_a(n), host_b(n), host_out(n, 0.0f), host_ref(n, 0.0f);
      fill_input(host_a, 1.0f);
      fill_input(host_b, 0.5f);
      cpu_reference(host_a, host_b, host_ref);

      float* device_a = nullptr;
      float* device_b = nullptr;
      float* device_out = nullptr;
      CHECK_CUDA(cudaMalloc(&device_a, bytes));
      CHECK_CUDA(cudaMalloc(&device_b, bytes));
      CHECK_CUDA(cudaMalloc(&device_out, bytes));
      CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

      const int threads = 256;
      const int blocks = (n + threads - 1) / threads;
      study_kernel<<<blocks, threads>>>(device_a, device_b, device_out, n);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaMemcpy(host_out.data(), device_out, bytes, cudaMemcpyDeviceToHost));

      int mismatches = 0;
      for (int i = 0; i < n; ++i) {{
        if (std::fabs(host_out[i] - host_ref[i]) > 1.0e-4f) {{
          ++mismatches;
        }}
      }}

      std::cout << "Blocks: " << blocks << ", Threads: " << threads << std::endl;
      std::cout << "Validation: " << (mismatches == 0 ? "PASS" : "UPDATE TEMPLATE LOGIC") << std::endl;

      CHECK_CUDA(cudaFree(device_a));
      CHECK_CUDA(cudaFree(device_b));
      CHECK_CUDA(cudaFree(device_out));
      return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }}

    // Suggested next steps:
    // 1. Replace study_kernel with the actual kernel for this algorithm.
    // 2. Expand cpu_reference to match the real computation.
    // 3. Add any extra buffers, atomics, scans, or shared-memory tiles you need.
    // 4. Test on tiny deterministic inputs first.
    // 5. Compare with CUDA libraries when the topic overlaps with one.
    """)

# Preserve existing batch-1 implementations while making them readable enough for review.
for example_dir in sorted(EXAMPLES.iterdir()):
    if not example_dir.is_dir():
        continue
    index = int(example_dir.name.split("_", 1)[0])
    title = title_from_readme(example_dir)
    write(example_dir / "README.md", readme_first20(index, title) if index <= 20 else readme_generic(index, title))
    if index > 20:
        write(example_dir / "main.cu", generic_code(index, title))

readme_path = ROOT / "README.md"
text = readme_path.read_text(encoding="utf-8")
text = text.replace("- 100 numbered example folders with CUDA study material", "- 100 numbered example folders with CUDA study material")
if "- Real CUDA implementations for examples `001-020`" not in text:
    text = text.replace("- Per-example study notes and implementation checklists", "- Real CUDA implementations for examples `001-020`\n- Per-example study notes and implementation checklists")
text = text.replace("- A scaffold generator script for the remaining template-heavy examples", "- A scaffold generator script for the remaining template-heavy examples")
readme_path.write_text(text, encoding="utf-8")
