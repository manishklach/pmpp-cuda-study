#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include "pmpp/benchmark.cuh"
#include "pmpp/cli.cuh"
#include "pmpp/compare.cuh"
#include "pmpp/cuda_check.cuh"
#include "pmpp/random_inputs.cuh"
#include "pmpp/report.cuh"

namespace {

constexpr const char *kExampleName = "111_warp-shuffle-reduction";
constexpr int kBlockSize = 256;
constexpr int kWarpCount = kBlockSize / 32;

__device__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

__global__ void warp_shuffle_reduction_kernel(const float *input, float *partials, int n) {
  __shared__ float warp_partials[kWarpCount];

  int global = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float thread_sum = 0.0f;
  for (int index = global; index < n; index += stride)
    thread_sum += input[index];

  float warp_sum = warp_reduce_sum(thread_sum);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0)
    warp_partials[warp] = warp_sum;
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < kWarpCount ? warp_partials[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0)
      partials[blockIdx.x] = block_sum;
  }
}

float cpu_reference(const std::vector<float> &input) {
  return std::accumulate(input.begin(), input.end(), 0.0f);
}

float run_gpu_once(const std::vector<float> &input) {
  const int n = static_cast<int>(input.size());
  const int blocks = std::max(1, (n + kBlockSize - 1) / kBlockSize);
  std::vector<float> partials(blocks, 0.0f);

  float *device_input = nullptr;
  float *device_partials = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_partials, blocks * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  warp_shuffle_reduction_kernel<<<blocks, kBlockSize>>>(device_input, device_partials, n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(partials.data(), device_partials, blocks * sizeof(float), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_partials));
  return std::accumulate(partials.begin(), partials.end(), 0.0f);
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  std::vector<float> input = pmpp::make_uniform_floats(options.size, options.seed, -1.0f, 1.0f);
  float cpu = cpu_reference(input);
  float gpu = run_gpu_once(input);
  pmpp::ValidationSummary summary = pmpp::compare_scalars(cpu, gpu, 1.0e-3);
  summary.notes = "Warp shuffles reduce values inside each warp before shared memory merges the warp partials.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int n = options.size;
  const int blocks = std::max(1, (n + kBlockSize - 1) / kBlockSize);
  std::vector<float> input = pmpp::make_uniform_floats(n, options.seed, -1.0f, 1.0f);
  float *device_input = nullptr;
  float *device_partials = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_partials, blocks * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    warp_shuffle_reduction_kernel<<<blocks, kBlockSize>>>(device_input, device_partials, n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps((static_cast<std::size_t>(n) + blocks) * sizeof(float), stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);
  stats.problem_label = "Input elements";
  stats.problem_size = static_cast<std::size_t>(n);

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_partials));
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
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Elements/sec");
  }

  return EXIT_SUCCESS;
}
