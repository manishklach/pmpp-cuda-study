from __future__ import annotations

from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]

TITLES = [
    "Hello World Kernel", "Vector Addition", "Vector Subtraction", "Scalar Vector Multiply",
    "Elementwise Array Square", "Elementwise Absolute Value", "SAXPY", "Copy Array Kernel",
    "Reverse Array", "Clamp Values To Range", "Threshold Binary Mask", "RGB To Grayscale",
    "Image Inversion", "Brightness Adjustment", "Contrast Adjustment", "1D Stencil",
    "2D Stencil", "Matrix Addition", "Matrix Transpose Naive", "Matrix Transpose With Shared Memory",
    "Dot Product", "L2 Norm", "Sum Reduction", "Max Reduction", "Min Reduction",
    "Prefix Sum Naive Scan", "Prefix Sum Work Efficient Scan", "Histogram Global Atomics",
    "Histogram Shared Memory", "Stream Compaction", "Gather", "Scatter", "Predicate Count",
    "Find First Match", "Parallel Even Odd Sort", "Bitonic Sort", "Odd Even Merge Sort",
    "Parallel Binary Search Over Sorted Chunks", "Merge Two Sorted Arrays", "Top K Selection",
    "Matrix Vector Multiply", "Naive Matrix Multiply", "Tiled Matrix Multiply", "Batched Matrix Multiply",
    "Convolution 1D", "Convolution 2D", "Separable Convolution", "Sobel Edge Detection",
    "Gaussian Blur", "Median Filter", "Box Filter With Shared Memory", "Sparse Matrix Vector Multiply CSR",
    "Sparse Matrix Dense Vector Multiply", "Jacobi Iteration", "Red Black Relaxation", "Power Iteration",
    "LU Factorization Sketch", "Cholesky Factorization", "QR Factorization Sketch", "FFT Based Convolution",
    "Image Resize Nearest Neighbor", "Image Resize Bilinear", "Template Matching", "Non Maximum Suppression",
    "Integral Image", "Canny Pipeline Stages", "Audio Gain And Mixing", "FIR Filter", "IIR Filter Sections",
    "Spectrogram With FFT", "Peak Detection", "Delta Encoding", "Run Length Encoding",
    "Parallel Base64 Or Hex Encode", "Block CRC Checksum", "Monte Carlo Pi", "Monte Carlo Option Pricing",
    "Random Walk Simulation", "N Body Naive", "N Body Tiled", "Lennard Jones Forces",
    "Heat Diffusion Grid", "Wave Equation Solver", "Lattice Boltzmann Step", "Game Of Life",
    "Boids Flocking", "Mandelbrot Renderer", "Julia Renderer", "Ray Sphere Tracer",
    "Path Tracing Diffuse Scene", "Parallel BFS", "Single Source Shortest Path", "PageRank",
    "Connected Components", "Union Find", "K Means Clustering", "DBSCAN Acceleration",
    "Neural Network Forward Pass", "MLP Backpropagation", "Multi GPU All Reduce Study",
]


def slug(index: int, title: str) -> str:
    return f"{index:03d}_{re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')}"


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
    common = {
        "Foundations": ["thread indexing", "bounds checks", "host-device memory flow"],
        "Parallel Patterns": ["work decomposition", "shared memory or atomics", "validation before tuning"],
        "Linear Algebra": ["data layout", "memory reuse", "correctness against a CPU reference"],
        "Image and Signal": ["2D or chunk indexing", "boundary handling", "pipeline composition"],
        "Simulation": ["state updates", "time stepping or sampling", "numerical checks"],
        "Graph and ML": ["irregular parallelism", "iteration strategy", "scalability planning"],
    }
    return common[track(index)]


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def md(text: str) -> str:
    return textwrap.dedent(text).strip()


def project_readme() -> str:
    lines = [
        "# PMPP CUDA Study With Templates",
        "",
        "A structured CUDA study repository with **100 examples** ordered from simple kernels to advanced PMPP-style workloads.",
        "",
        "## Included",
        "",
        "- 100 numbered example folders with CUDA starter code",
        "- Per-example study notes and implementation checklists",
        "- A 5-batch GitHub publishing plan for `001-020` through `081-100`",
        "- A generator script so the structure stays consistent",
        "",
        "## Quick Start",
        "",
        "```powershell",
        "cd examples\\001_hello-world-kernel",
        "nvcc -std=c++17 -O2 main.cu -o example.exe",
        ".\\example.exe",
        "```",
        "",
        "## Example Index",
        "",
        "| # | Example | Track | Difficulty | Link |",
        "|---|---|---|---|---|",
    ]
    for i, title in enumerate(TITLES, start=1):
        lines.append(
            f"| {i:03d} | {title} | {track(i)} | {difficulty(i)} | [Open](examples/{slug(i, title)}/README.md) |"
        )
    return "\n".join(lines)


def conventions_doc() -> str:
    return "\n".join([
        "# Example Conventions",
        "",
        "Each example contains:",
        "",
        "- `README.md` with goal, PMPP ideas, build/run steps, and study prompts",
        "- `main.cu` with a CUDA scaffold you can compile and extend",
        "",
        "Labels:",
        "",
        "- `Reference-friendly`: earlier examples designed to be easy to build and inspect",
        "- `Guided template`: later examples that emphasize structure, TODOs, and study direction",
    ])


def publish_doc() -> str:
    return "\n".join([
        "# Publish In 5 Batches",
        "",
        "Recommended commit sequence:",
        "",
        "1. `001-020`",
        "2. `021-040`",
        "3. `041-060`",
        "4. `061-080`",
        "5. `081-100`",
        "",
        "Suggested commit messages:",
        "",
        "- `feat: add PMPP CUDA study examples 001-020`",
        "- `feat: add PMPP CUDA study examples 021-040`",
        "- `feat: add PMPP CUDA study examples 041-060`",
        "- `feat: add PMPP CUDA study examples 061-080`",
        "- `feat: add PMPP CUDA study examples 081-100`",
    ])


def cmake_file() -> str:
    return textwrap.dedent(
        """
        cmake_minimum_required(VERSION 3.24)
        project(pmpp_cuda_study LANGUAGES CXX CUDA)

        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_CUDA_STANDARD 17)

        file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/examples/*/main.cu")
        foreach(EXAMPLE_SOURCE IN LISTS EXAMPLE_SOURCES)
          get_filename_component(EXAMPLE_DIR "${EXAMPLE_SOURCE}" DIRECTORY)
          get_filename_component(EXAMPLE_NAME "${EXAMPLE_DIR}" NAME)
          add_executable(${EXAMPLE_NAME} "${EXAMPLE_SOURCE}")
          set_target_properties(${EXAMPLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        endforeach()
        """
    ).strip()


def example_readme(index: int, title: str) -> str:
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
    lines.extend(f"- {item}" for item in concepts(index))
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


def example_code(index: int, title: str) -> str:
    notes = "\n".join(f"// - Study focus: {item}" for item in concepts(index))
    suffix = ""
    if index > 20:
        suffix = textwrap.dedent(
            """

            // Suggested next steps:
            // 1. Replace study_kernel with the actual kernel for this algorithm.
            // 2. Expand cpu_reference to match the real computation.
            // 3. Add any extra buffers, atomics, scans, or shared-memory tiles you need.
            // 4. Test on tiny deterministic inputs first.
            // 5. Compare with CUDA libraries when the topic overlaps with one.
            """
        ).rstrip()
    return textwrap.dedent(
        f"""
        // Example {index:03d}: {title}
        // Track: {track(index)}
        // Difficulty: {difficulty(index)}
        // Status: {status(index)}
        {notes}

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
          std::cout << "Running {slug(index, title)}" << std::endl;

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
        }}{suffix}
        """
    ).strip() + "\n"


def main() -> None:
    write(ROOT / "README.md", project_readme())
    write(ROOT / "docs" / "example-conventions.md", conventions_doc())
    write(ROOT / "docs" / "publish-in-5-batches.md", publish_doc())
    write(ROOT / "CMakeLists.txt", cmake_file())
    write(ROOT / ".gitignore", "build/\n*.exe\n")
    for index, title in enumerate(TITLES, start=1):
        base = ROOT / "examples" / slug(index, title)
        write(base / "README.md", example_readme(index, title))
        write(base / "main.cu", example_code(index, title))


if __name__ == "__main__":
    main()
