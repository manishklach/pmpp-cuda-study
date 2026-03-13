#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "pmpp/benchmark.cuh"
#include "pmpp/cli.cuh"
#include "pmpp/compare.cuh"
#include "pmpp/cuda_check.cuh"
#include "pmpp/report.cuh"

namespace {
constexpr const char *kExampleName = "048_sobel-edge-detection";
__global__ void sobel_kernel(const float *input, float *output, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x > 0 && x + 1 < w && y > 0 && y + 1 < h) {
    int idx = y * w + x;
    float gx = -input[(y - 1) * w + (x - 1)] + input[(y - 1) * w + (x + 1)] -
               2 * input[y * w + (x - 1)] + 2 * input[y * w + (x + 1)] -
               input[(y + 1) * w + (x - 1)] + input[(y + 1) * w + (x + 1)];
    float gy = -input[(y - 1) * w + (x - 1)] - 2 * input[(y - 1) * w + x] -
               input[(y - 1) * w + (x + 1)] + input[(y + 1) * w + (x - 1)] +
               2 * input[(y + 1) * w + x] + input[(y + 1) * w + (x + 1)];
    output[idx] = sqrtf(gx * gx + gy * gy);
  }
}
}

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  int w = options.size;
  int h = options.size;
  std::vector<float> input(w * h, 0.0f), cpu(w * h, 0.0f), gpu(w * h, 0.0f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      input[y * w + x] = (x < w / 2 ? 0.0f : 10.0f);
  for (int y = 1; y < h - 1; ++y)
    for (int x = 1; x < w - 1; ++x) {
      float gx = -input[(y - 1) * w + (x - 1)] + input[(y - 1) * w + (x + 1)] -
                 2 * input[y * w + (x - 1)] + 2 * input[y * w + (x + 1)] -
                 input[(y + 1) * w + (x - 1)] + input[(y + 1) * w + (x + 1)];
      float gy = -input[(y - 1) * w + (x - 1)] - 2 * input[(y - 1) * w + x] -
                 input[(y - 1) * w + (x + 1)] + input[(y + 1) * w + (x - 1)] +
                 2 * input[(y + 1) * w + x] + input[(y + 1) * w + (x + 1)];
      cpu[y * w + x] = std::sqrt(gx * gx + gy * gy);
    }
  if (options.check) {
    float *di = nullptr, *dout = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&di, input.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dout, gpu.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMemcpy(di, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    dim3 t(16, 16), bl((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
    sobel_kernel<<<bl, t>>>(di, dout, w, h);
    PMPP_CUDA_KERNEL_CHECK();
    PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), dout, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-4f);
    summary.notes = "Sobel computes gradient magnitude from horizontal and vertical edge filters.";
    pmpp::print_validation_report(kExampleName, summary);
    PMPP_CUDA_CHECK(cudaFree(di));
    PMPP_CUDA_CHECK(cudaFree(dout));
    if (!summary.ok)
      return EXIT_FAILURE;
  }
  if (options.bench) {
    float *di = nullptr, *dout = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&di, input.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dout, gpu.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMemcpy(di, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    dim3 t(16, 16), bl((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
    pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
      sobel_kernel<<<bl, t>>>(di, dout, w, h);
      PMPP_CUDA_KERNEL_CHECK();
    });
    stats.bandwidth_gbps = pmpp::bandwidth_gbps((input.size() + gpu.size()) * sizeof(float), stats.avg_ms);
    stats.throughput = pmpp::elements_per_second(gpu.size(), stats.avg_ms);
    if (!options.verify)
      std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Pixels/sec");
    PMPP_CUDA_CHECK(cudaFree(di));
    PMPP_CUDA_CHECK(cudaFree(dout));
  }
  return EXIT_SUCCESS;
}
