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

constexpr const char *kExampleName = "137_heat-diffusion-tiled-2d";
constexpr int kTile = 16;
constexpr float kAlpha = 0.15f;

__global__ void heat_diffusion_kernel(const float *input, float *output, int width, int height) {
  __shared__ float tile[kTile + 2][kTile + 2];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * kTile + tx;
  int y = blockIdx.y * kTile + ty;
  int sx = tx + 1;
  int sy = ty + 1;

  auto read = [&](int gx, int gy) {
    gx = max(0, min(gx, width - 1));
    gy = max(0, min(gy, height - 1));
    return input[gy * width + gx];
  };

  tile[sy][sx] = read(x, y);
  if (tx == 0)
    tile[sy][0] = read(x - 1, y);
  if (tx == kTile - 1)
    tile[sy][kTile + 1] = read(x + 1, y);
  if (ty == 0)
    tile[0][sx] = read(x, y - 1);
  if (ty == kTile - 1)
    tile[kTile + 1][sx] = read(x, y + 1);
  __syncthreads();

  if (x < width && y < height) {
    float center = tile[sy][sx];
    float laplacian = tile[sy - 1][sx] + tile[sy + 1][sx] + tile[sy][sx - 1] + tile[sy][sx + 1] - 4.0f * center;
    output[y * width + x] = center + kAlpha * laplacian;
  }
}

std::vector<float> cpu_reference(const std::vector<float> &input, int width, int height) {
  auto read = [&](int x, int y) {
    x = std::max(0, std::min(x, width - 1));
    y = std::max(0, std::min(y, height - 1));
    return input[y * width + x];
  };
  std::vector<float> output(input.size(), 0.0f);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      float center = read(x, y);
      float laplacian = read(x, y - 1) + read(x, y + 1) + read(x - 1, y) + read(x + 1, y) - 4.0f * center;
      output[y * width + x] = center + kAlpha * laplacian;
    }
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int width = std::max(64, options.size);
  const int height = std::max(64, options.size - options.size / 5);
  const std::size_t count = static_cast<std::size_t>(width) * height;
  std::vector<float> input = pmpp::make_uniform_floats(count, options.seed, 0.0f, 1.0f);
  std::vector<float> cpu = cpu_reference(input, width, height);
  std::vector<float> gpu(count, 0.0f);

  float *device_input = nullptr;
  float *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, count * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, count * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((width + kTile - 1) / kTile, (height + kTile - 1) / kTile);
  heat_diffusion_kernel<<<blocks, threads>>>(device_input, device_output, width, height);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, count * sizeof(float), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-5f);
  summary.notes = "This example performs one tiled heat-diffusion step with shared-memory halo staging.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int width = std::max(64, options.size);
  const int height = std::max(64, options.size - options.size / 5);
  const std::size_t count = static_cast<std::size_t>(width) * height;
  std::vector<float> input = pmpp::make_uniform_floats(count, options.seed, 0.0f, 1.0f);
  float *device_input = nullptr;
  float *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, count * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, count * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((width + kTile - 1) / kTile, (height + kTile - 1) / kTile);
  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    heat_diffusion_kernel<<<blocks, threads>>>(device_input, device_output, width, height);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(count * sizeof(float) * 2, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(count, stats.avg_ms);
  stats.problem_label = "Grid points";
  stats.problem_size = count;

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));
  return stats;
}

}  // namespace

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  if (options.check) {
    pmpp::ValidationSummary summary = run_check(options);
    pmpp::print_validation_report(kExampleName, summary);
    if (!summary.ok)
      return EXIT_FAILURE;
  }
  if (options.bench) {
    if (!options.verify)
      std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::BenchmarkStats stats = run_bench(options);
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Points/sec");
  }
  return EXIT_SUCCESS;
}
