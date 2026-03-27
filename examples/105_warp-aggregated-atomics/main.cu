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

constexpr const char *kExampleName = "105_warp-aggregated-atomics";

__global__ void count_positive_naive_kernel(const int *input, int *count, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    if (input[i] > 0)
      atomicAdd(count, 1);
  }
}

__global__ void count_positive_warp_aggregated_kernel(const int *input, int *count, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride) {
    bool active = input[i] > 0;
    unsigned int mask = __ballot_sync(0xffffffffu, active);

    // Only one lane per warp performs the atomic increment, reserving all positive hits from the
    // warp at once. That reduces atomic traffic when many lanes update the same counter.
    if ((threadIdx.x & 31) == (static_cast<int>(__ffs(mask)) - 1) && mask != 0)
      atomicAdd(count, __popc(mask));
  }
}

int cpu_reference(const std::vector<int> &input) {
  int total = 0;
  for (int value : input)
    total += value > 0 ? 1 : 0;
  return total;
}

int run_kernel(const std::vector<int> &input, bool use_warp_aggregation, int block_size) {
  const int n = static_cast<int>(input.size());
  const int blocks = std::max(1, (n + block_size - 1) / block_size);
  int *device_input = nullptr;
  int *device_count = nullptr;
  int count = 0;

  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_count, sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemset(device_count, 0, sizeof(int)));

  if (use_warp_aggregation)
    count_positive_warp_aggregated_kernel<<<blocks, block_size>>>(device_input, device_count, n);
  else
    count_positive_naive_kernel<<<blocks, block_size>>>(device_input, device_count, n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(&count, device_count, sizeof(int), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_count));
  return count;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int block_size = 256;
  std::vector<int> input = pmpp::make_uniform_ints(options.size, options.seed, -3, 3);
  int cpu = cpu_reference(input);
  int naive = run_kernel(input, false, block_size);
  int aggregated = run_kernel(input, true, block_size);

  pmpp::ValidationSummary naive_summary = pmpp::compare_scalars(cpu, naive, 0.0);
  pmpp::ValidationSummary aggregated_summary = pmpp::compare_scalars(cpu, aggregated, 0.0);
  pmpp::ValidationSummary summary = aggregated_summary;
  summary.ok = naive_summary.ok && aggregated_summary.ok;
  summary.mismatch_count = naive_summary.mismatch_count + aggregated_summary.mismatch_count;
  summary.max_abs_error = std::max(naive_summary.max_abs_error, aggregated_summary.max_abs_error);
  summary.notes = "Validated both the naive atomic counter and the warp-aggregated variant.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int block_size = 256;
  const int n = options.size;
  const int blocks = std::max(1, (n + block_size - 1) / block_size);
  std::vector<int> input = pmpp::make_uniform_ints(n, options.seed, -3, 3);
  int *device_input = nullptr;
  int *device_count = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_count, sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    PMPP_CUDA_CHECK(cudaMemset(device_count, 0, sizeof(int)));
    count_positive_warp_aggregated_kernel<<<blocks, block_size>>>(device_input, device_count, n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(static_cast<std::size_t>(n) * sizeof(int), stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);
  stats.problem_label = "Input elements";
  stats.problem_size = static_cast<std::size_t>(n);

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_count));
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
