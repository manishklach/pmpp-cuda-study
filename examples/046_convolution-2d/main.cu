#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "pmpp/benchmark.cuh"
#include "pmpp/cli.cuh"
#include "pmpp/compare.cuh"
#include "pmpp/cuda_check.cuh"
#include "pmpp/random_inputs.cuh"
#include "pmpp/report.cuh"

namespace {
constexpr const char *kExampleName = "046_convolution-2d";
constexpr int kRadius = 1;
__global__ void conv2d_kernel(const float *input, const float *kernel, float *output, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    float sum = 0.0f;
    for (int ky = -kRadius; ky <= kRadius; ++ky)
      for (int kx = -kRadius; kx <= kRadius; ++kx) {
        int sx = min(max(x + kx, 0), w - 1);
        int sy = min(max(y + ky, 0), h - 1);
        sum += input[sy * w + sx] * kernel[(ky + kRadius) * 3 + (kx + kRadius)];
      }
    output[y * w + x] = sum;
  }
}
}

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  int w = options.size, h = options.size;
  std::vector<float> input = pmpp::make_uniform_floats(w * h, options.seed, 0.0f, 1.0f);
  std::vector<float> kernel = {0.0f, 0.125f, 0.0f, 0.125f, 0.5f, 0.125f, 0.0f, 0.125f, 0.0f};
  std::vector<float> cpu(w * h, 0.0f), gpu(w * h, 0.0f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int ky = -kRadius; ky <= kRadius; ++ky)
        for (int kx = -kRadius; kx <= kRadius; ++kx) {
          int sx = std::clamp(x + kx, 0, w - 1);
          int sy = std::clamp(y + ky, 0, h - 1);
          cpu[y * w + x] += input[sy * w + sx] * kernel[(ky + kRadius) * 3 + (kx + kRadius)];
        }

  if (options.check) {
    float *di = nullptr, *dk = nullptr, *dout = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&di, input.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dk, kernel.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dout, gpu.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMemcpy(di, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    PMPP_CUDA_CHECK(cudaMemcpy(dk, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice));
    dim3 t(16, 16), bl((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
    conv2d_kernel<<<bl, t>>>(di, dk, dout, w, h);
    PMPP_CUDA_KERNEL_CHECK();
    PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), dout, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-5f);
    summary.notes = "This direct 2D convolution is the baseline for separable or tiled variants.";
    pmpp::print_validation_report(kExampleName, summary);
    PMPP_CUDA_CHECK(cudaFree(di)); PMPP_CUDA_CHECK(cudaFree(dk)); PMPP_CUDA_CHECK(cudaFree(dout));
    if (!summary.ok) return EXIT_FAILURE;
  }
  if (options.bench) {
    float *di = nullptr, *dk = nullptr, *dout = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&di, input.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dk, kernel.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dout, gpu.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMemcpy(di, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    PMPP_CUDA_CHECK(cudaMemcpy(dk, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice));
    dim3 t(16, 16), bl((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
    pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
      conv2d_kernel<<<bl, t>>>(di, dk, dout, w, h);
      PMPP_CUDA_KERNEL_CHECK();
    });
    stats.bandwidth_gbps = pmpp::bandwidth_gbps((input.size() + gpu.size() + kernel.size()) * sizeof(float), stats.avg_ms);
    stats.throughput = pmpp::elements_per_second(gpu.size(), stats.avg_ms);
    if (!options.verify) std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Pixels/sec");
    PMPP_CUDA_CHECK(cudaFree(di)); PMPP_CUDA_CHECK(cudaFree(dk)); PMPP_CUDA_CHECK(cudaFree(dout));
  }
  return EXIT_SUCCESS;
}
