#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
constexpr const char *kExampleName = "061_image-resize-nearest-neighbor";
__global__ void resize_nearest_kernel(const float *src, int src_width, int src_height, float *dst,
                                      int dst_width, int dst_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_width || y >= dst_height) return;
  float scale_x = static_cast<float>(src_width) / static_cast<float>(dst_width);
  float scale_y = static_cast<float>(src_height) / static_cast<float>(dst_height);
  int src_x = min(static_cast<int>(x * scale_x), src_width - 1);
  int src_y = min(static_cast<int>(y * scale_y), src_height - 1);
  dst[y * dst_width + x] = src[src_y * src_width + src_x];
}
}

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  int src_width = std::max(4, options.size / 2);
  int src_height = std::max(4, options.size / 3);
  int dst_width = options.size;
  int dst_height = std::max(4, options.size - options.size / 4);
  std::vector<float> src = pmpp::make_uniform_floats(src_width * src_height, options.seed, 0.0f, 10.0f);
  std::vector<float> cpu(dst_width * dst_height, 0.0f), gpu(dst_width * dst_height, 0.0f);
  for (int y = 0; y < dst_height; ++y)
    for (int x = 0; x < dst_width; ++x) {
      int sx = std::min(static_cast<int>(x * (static_cast<float>(src_width) / dst_width)), src_width - 1);
      int sy = std::min(static_cast<int>(y * (static_cast<float>(src_height) / dst_height)), src_height - 1);
      cpu[y * dst_width + x] = src[sy * src_width + sx];
    }
  if (options.check) {
    float *dsrc = nullptr, *ddst = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&dsrc, src.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&ddst, gpu.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMemcpy(dsrc, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    dim3 threads(16, 16), blocks((dst_width + 15) / 16, (dst_height + 15) / 16);
    resize_nearest_kernel<<<blocks, threads>>>(dsrc, src_width, src_height, ddst, dst_width, dst_height);
    PMPP_CUDA_KERNEL_CHECK();
    PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), ddst, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-6f);
    summary.notes = "Nearest-neighbor resize is a clean 2D remapping example with no interpolation arithmetic.";
    pmpp::print_validation_report(kExampleName, summary);
    PMPP_CUDA_CHECK(cudaFree(dsrc)); PMPP_CUDA_CHECK(cudaFree(ddst));
    if (!summary.ok) return EXIT_FAILURE;
  }
  if (options.bench) {
    float *dsrc = nullptr, *ddst = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&dsrc, src.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&ddst, gpu.size() * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMemcpy(dsrc, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    dim3 threads(16, 16), blocks((dst_width + 15) / 16, (dst_height + 15) / 16);
    pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
      resize_nearest_kernel<<<blocks, threads>>>(dsrc, src_width, src_height, ddst, dst_width, dst_height);
      PMPP_CUDA_KERNEL_CHECK();
    });
    stats.bandwidth_gbps = pmpp::bandwidth_gbps((src.size() + gpu.size()) * sizeof(float), stats.avg_ms);
    stats.throughput = pmpp::elements_per_second(gpu.size(), stats.avg_ms);
    if (!options.verify) std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Pixels/sec");
    PMPP_CUDA_CHECK(cudaFree(dsrc)); PMPP_CUDA_CHECK(cudaFree(ddst));
  }
  return EXIT_SUCCESS;
}
