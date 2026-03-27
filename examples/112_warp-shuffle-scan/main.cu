#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "pmpp/benchmark.cuh"
#include "pmpp/cli.cuh"
#include "pmpp/compare.cuh"
#include "pmpp/cuda_check.cuh"
#include "pmpp/report.cuh"

namespace {

constexpr const char *kExampleName = "112_warp-shuffle-scan";
constexpr int kMaxThreads = 256;
constexpr int kMaxWarps = kMaxThreads / 32;

int round_up_to_warp(int n) {
  return ((n + 31) / 32) * 32;
}

__device__ int warp_inclusive_scan(int value) {
  for (int offset = 1; offset < 32; offset <<= 1) {
    int neighbor = __shfl_up_sync(0xffffffffu, value, offset);
    if ((threadIdx.x & 31) >= offset)
      value += neighbor;
  }
  return value;
}

__global__ void warp_shuffle_scan_kernel(const int *input, int *output, int n) {
  __shared__ int warp_totals[kMaxWarps];
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warp_count = blockDim.x / 32;

  int value = tid < n ? input[tid] : 0;
  int scan = warp_inclusive_scan(value);

  if (lane == 31)
    warp_totals[warp] = scan;
  __syncthreads();

  if (warp == 0) {
    int warp_value = lane < warp_count ? warp_totals[lane] : 0;
    int warp_scan = warp_inclusive_scan(warp_value);
    if (lane < warp_count)
      warp_totals[lane] = warp_scan;
  }
  __syncthreads();

  int carry = warp == 0 ? 0 : warp_totals[warp - 1];
  if (tid < n)
    output[tid] = scan + carry;
}

std::vector<int> make_input(int n) {
  std::vector<int> input(n, 0);
  for (int i = 0; i < n; ++i)
    input[i] = (i % 5) + 1;
  return input;
}

std::vector<int> cpu_reference(const std::vector<int> &input) {
  std::vector<int> output(input.size(), 0);
  for (std::size_t i = 0; i < input.size(); ++i)
    output[i] = input[i] + (i == 0 ? 0 : output[i - 1]);
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int n = std::max(32, std::min(options.size, kMaxThreads));
  const int threads = round_up_to_warp(n);
  std::vector<int> input = make_input(n);
  std::vector<int> cpu = cpu_reference(input);
  std::vector<int> gpu(n, 0);

  int *device_input = nullptr;
  int *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  warp_shuffle_scan_kernel<<<1, threads>>>(device_input, device_output, n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, n * sizeof(int), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu);
  summary.notes = "This scan uses warp-local shuffles plus scanned warp totals to build a block-wide prefix sum.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int n = std::max(32, std::min(options.size, kMaxThreads));
  const int threads = round_up_to_warp(n);
  std::vector<int> input = make_input(n);

  int *device_input = nullptr;
  int *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    warp_shuffle_scan_kernel<<<1, threads>>>(device_input, device_output, n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(static_cast<std::size_t>(n) * sizeof(int) * 2, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);
  stats.problem_label = "Scanned elements";
  stats.problem_size = static_cast<std::size_t>(n);

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
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Elements/sec");
  }
  return EXIT_SUCCESS;
}
