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
constexpr const char *kExampleName = "050_median-filter";
__device__ void sort9(float *vals) {
  for (int i = 0; i < 9; ++i)
    for (int j = i + 1; j < 9; ++j)
      if (vals[j] < vals[i]) {
        float t = vals[i];
        vals[i] = vals[j];
        vals[j] = t;
      }
}
__global__ void median3x3_kernel(const float *input, float *output, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    float vals[9];
    int p = 0;
    for (int ky = -1; ky <= 1; ++ky)
      for (int kx = -1; kx <= 1; ++kx) {
        int sx = min(max(x + kx, 0), w - 1);
        int sy = min(max(y + ky, 0), h - 1);
        vals[p++] = input[sy * w + sx];
      }
    sort9(vals);
    output[y * w + x] = vals[4];
  }
}
}

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  int w = options.size, h = options.size;
  std::vector<float> input = pmpp::make_uniform_floats(w * h, options.seed, 0.0f, 10.0f);
  if (w * h > 10) input[10] = 99.0f;
  std::vector<float> cpu(w * h, 0.0f), gpu(w * h, 0.0f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float vals[9];
      int p = 0;
      for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx) {
          int sx = std::clamp(x + kx, 0, w - 1);
          int sy = std::clamp(y + ky, 0, h - 1);
          vals[p++] = input[sy * w + sx];
        }
      std::sort(vals, vals + 9);
      cpu[y * w + x] = vals[4];
    }
  if (options.check) {
    float *di = nullptr, *dout = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&di, input.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dout, gpu.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMemcpy(di, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    dim3 t(16, 16), bl((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
    median3x3_kernel<<<bl, t>>>(di, dout, w, h);
    PMPP_CUDA_KERNEL_CHECK();
    PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), dout, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-5f);
    summary.notes = "Median filtering is nonlinear, so it is a good contrast against convolution-based filters.";
    pmpp::print_validation_report(kExampleName, summary);
    PMPP_CUDA_CHECK(cudaFree(di)); PMPP_CUDA_CHECK(cudaFree(dout));
    if (!summary.ok) return EXIT_FAILURE;
  }
  if (options.bench) {
    float *di = nullptr, *dout = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&di, input.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dout, gpu.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMemcpy(di, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    dim3 t(16, 16), bl((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
    pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
      median3x3_kernel<<<bl, t>>>(di, dout, w, h);
      PMPP_CUDA_KERNEL_CHECK();
    });
    stats.bandwidth_gbps = pmpp::bandwidth_gbps((input.size() + gpu.size()) * sizeof(float), stats.avg_ms);
    stats.throughput = pmpp::elements_per_second(gpu.size(), stats.avg_ms);
    if (!options.verify) std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Pixels/sec");
    PMPP_CUDA_CHECK(cudaFree(di)); PMPP_CUDA_CHECK(cudaFree(dout));
  }
  return EXIT_SUCCESS;
}
