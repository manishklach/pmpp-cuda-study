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
#include "pmpp/report.cuh"

namespace {

constexpr const char *kExampleName = "131_sobel-filter-optimized";
constexpr int kTile = 16;

__global__ void sobel_tiled_kernel(const float *input, float *output, int width, int height) {
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
  if (tx == 0 && ty == 0)
    tile[0][0] = read(x - 1, y - 1);
  if (tx == kTile - 1 && ty == 0)
    tile[0][kTile + 1] = read(x + 1, y - 1);
  if (tx == 0 && ty == kTile - 1)
    tile[kTile + 1][0] = read(x - 1, y + 1);
  if (tx == kTile - 1 && ty == kTile - 1)
    tile[kTile + 1][kTile + 1] = read(x + 1, y + 1);
  __syncthreads();

  if (x < width && y < height) {
    float gx = -tile[sy - 1][sx - 1] - 2.0f * tile[sy][sx - 1] - tile[sy + 1][sx - 1] +
               tile[sy - 1][sx + 1] + 2.0f * tile[sy][sx + 1] + tile[sy + 1][sx + 1];
    float gy = tile[sy - 1][sx - 1] + 2.0f * tile[sy - 1][sx] + tile[sy - 1][sx + 1] -
               tile[sy + 1][sx - 1] - 2.0f * tile[sy + 1][sx] - tile[sy + 1][sx + 1];
    output[y * width + x] = fabsf(gx) + fabsf(gy);
  }
}

std::vector<float> make_image(int width, int height) {
  std::vector<float> image(static_cast<std::size_t>(width) * height, 0.0f);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      image[y * width + x] = 0.2f * x + 0.3f * y + ((x / 8 + y / 8) % 2 ? 3.0f : -1.0f);
  return image;
}

std::vector<float> cpu_reference(const std::vector<float> &input, int width, int height) {
  auto read = [&](int x, int y) {
    x = std::max(0, std::min(x, width - 1));
    y = std::max(0, std::min(y, height - 1));
    return input[y * width + x];
  };
  std::vector<float> output(input.size(), 0.0f);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float gx = -read(x - 1, y - 1) - 2.0f * read(x - 1, y) - read(x - 1, y + 1) +
                 read(x + 1, y - 1) + 2.0f * read(x + 1, y) + read(x + 1, y + 1);
      float gy = read(x - 1, y - 1) + 2.0f * read(x, y - 1) + read(x + 1, y - 1) -
                 read(x - 1, y + 1) - 2.0f * read(x, y + 1) - read(x + 1, y + 1);
      output[y * width + x] = std::fabs(gx) + std::fabs(gy);
    }
  }
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int width = std::max(64, options.size);
  const int height = std::max(64, options.size - options.size / 5);
  const std::size_t count = static_cast<std::size_t>(width) * height;
  std::vector<float> input = make_image(width, height);
  std::vector<float> cpu = cpu_reference(input, width, height);
  std::vector<float> gpu(count, 0.0f);

  float *device_input = nullptr;
  float *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, count * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, count * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((width + kTile - 1) / kTile, (height + kTile - 1) / kTile);
  sobel_tiled_kernel<<<blocks, threads>>>(device_input, device_output, width, height);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, count * sizeof(float), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-5f);
  summary.notes = "This Sobel variant stages a tile plus halo so each block reuses neighboring pixels from shared memory.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int width = std::max(64, options.size);
  const int height = std::max(64, options.size - options.size / 5);
  const std::size_t count = static_cast<std::size_t>(width) * height;
  std::vector<float> input = make_image(width, height);
  float *device_input = nullptr;
  float *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, count * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, count * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((width + kTile - 1) / kTile, (height + kTile - 1) / kTile);
  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    sobel_tiled_kernel<<<blocks, threads>>>(device_input, device_output, width, height);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(count * sizeof(float) * 2, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(count, stats.avg_ms);
  stats.problem_label = "Pixels";
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
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Pixels/sec");
  }
  return EXIT_SUCCESS;
}
