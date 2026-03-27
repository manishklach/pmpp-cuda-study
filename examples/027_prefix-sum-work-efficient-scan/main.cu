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

constexpr const char *kExampleName = "027_prefix-sum-work-efficient-scan";
constexpr int kMaxScanElements = 1024;

int clamp_problem_size(int requested) {
  return std::max(2, std::min(requested, kMaxScanElements));
}

int next_power_of_two(int value) {
  int power = 1;
  while (power < value)
    power <<= 1;
  return power;
}

__global__ void blelloch_inclusive_scan_kernel(const int *input, int *output, int n, int padded_n) {
  __shared__ int scratch[kMaxScanElements];
  int tid = threadIdx.x;

  scratch[tid] = tid < n ? input[tid] : 0;
  __syncthreads();

  // Up-sweep builds a reduction tree in shared memory. The active indices spread apart,
  // so threads cooperate on fewer totals each level while preserving O(n) total work.
  for (int stride = 1; stride < padded_n; stride <<= 1) {
    int index = ((tid + 1) * stride * 2) - 1;
    if (index < padded_n)
      scratch[index] += scratch[index - stride];
    __syncthreads();
  }

  if (tid == 0)
    scratch[padded_n - 1] = 0;
  __syncthreads();

  // Down-sweep propagates prefix totals back to every element. Synchronization is required
  // after each swap so no thread reads partially updated state from another thread.
  for (int stride = padded_n >> 1; stride > 0; stride >>= 1) {
    int index = ((tid + 1) * stride * 2) - 1;
    if (index < padded_n) {
      int left = scratch[index - stride];
      scratch[index - stride] = scratch[index];
      scratch[index] += left;
    }
    __syncthreads();
  }

  if (tid < n)
    output[tid] = scratch[tid] + input[tid];
}

std::vector<int> cpu_reference(const std::vector<int> &input) {
  std::vector<int> output(input.size(), 0);
  for (std::size_t i = 0; i < input.size(); ++i)
    output[i] = input[i] + (i == 0 ? 0 : output[i - 1]);
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int n = clamp_problem_size(options.size);
  const int padded_n = next_power_of_two(n);
  std::vector<int> input = pmpp::make_uniform_ints(n, options.seed, 0, 5);
  std::vector<int> cpu = cpu_reference(input);
  std::vector<int> gpu(n, 0);

  int *device_input = nullptr;
  int *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  blelloch_inclusive_scan_kernel<<<1, padded_n>>>(device_input, device_output, n, padded_n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, n * sizeof(int), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu);
  summary.notes =
      "Work-efficient scan uses an up-sweep/down-sweep tree, so it does O(n) total adds.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int n = clamp_problem_size(options.size);
  const int padded_n = next_power_of_two(n);
  std::vector<int> input = pmpp::make_uniform_ints(n, options.seed, 0, 5);

  int *device_input = nullptr;
  int *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    blelloch_inclusive_scan_kernel<<<1, padded_n>>>(device_input, device_output, n, padded_n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(static_cast<std::size_t>(n) * sizeof(int) * 2,
                                              stats.avg_ms);
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
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters,
                                 "Elements/sec");
  }

  return EXIT_SUCCESS;
}
