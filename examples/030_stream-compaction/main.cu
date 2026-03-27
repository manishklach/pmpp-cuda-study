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

constexpr const char *kExampleName = "030_stream-compaction";
constexpr int kMaxCompactElements = 1024;

int clamp_problem_size(int requested) {
  return std::max(2, std::min(requested, kMaxCompactElements));
}

int next_power_of_two(int value) {
  int power = 1;
  while (power < value)
    power <<= 1;
  return power;
}

std::vector<int> make_input(int n, unsigned int seed) {
  std::vector<int> values(n, 0);
  for (int i = 0; i < n; ++i)
    values[i] = ((i * 5 + static_cast<int>(seed)) % 11) - 5;
  return values;
}

__global__ void compact_atomic_kernel(const int *input, int *output, int *count, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride) {
    if (input[i] > 0) {
      // This baseline is simple, but every surviving item contends on one global counter and
      // the resulting output order depends on which thread reaches the atomic first.
      int slot = atomicAdd(count, 1);
      output[slot] = input[i];
    }
  }
}

__global__ void compact_stable_scan_kernel(const int *input, int *output, int *count, int n,
                                           int padded_n) {
  __shared__ int flags[kMaxCompactElements];
  int tid = threadIdx.x;
  int value = tid < n ? input[tid] : 0;
  int flag = (tid < n && value > 0) ? 1 : 0;

  flags[tid] = flag;
  __syncthreads();

  for (int stride = 1; stride < padded_n; stride <<= 1) {
    int index = ((tid + 1) * stride * 2) - 1;
    if (index < padded_n)
      flags[index] += flags[index - stride];
    __syncthreads();
  }

  if (tid == 0)
    flags[padded_n - 1] = 0;
  __syncthreads();

  for (int stride = padded_n >> 1; stride > 0; stride >>= 1) {
    int index = ((tid + 1) * stride * 2) - 1;
    if (index < padded_n) {
      int left = flags[index - stride];
      flags[index - stride] = flags[index];
      flags[index] += left;
    }
    __syncthreads();
  }

  // The exclusive scan gives each surviving element its final output slot, so this version
  // preserves input order and removes the global atomic hot spot. The tradeoff is extra shared
  // memory traffic and a synchronization-heavy scan phase.
  if (tid < n && flag)
    output[flags[tid]] = value;

  if (tid == n - 1)
    *count = flags[tid] + flag;
}

std::vector<int> cpu_reference(const std::vector<int> &input) {
  std::vector<int> output;
  for (int value : input)
    if (value > 0)
      output.push_back(value);
  return output;
}

std::vector<int> run_atomic_compaction(const std::vector<int> &input) {
  const int n = static_cast<int>(input.size());
  const int threads = 256;
  const int blocks = std::max(1, (n + threads - 1) / threads);
  std::vector<int> gpu(n, 0);

  int *device_input = nullptr;
  int *device_output = nullptr;
  int *device_count = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_count, sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemset(device_count, 0, sizeof(int)));

  compact_atomic_kernel<<<blocks, threads>>>(device_input, device_output, device_count, n);
  PMPP_CUDA_KERNEL_CHECK();

  int count = 0;
  PMPP_CUDA_CHECK(cudaMemcpy(&count, device_count, sizeof(int), cudaMemcpyDeviceToHost));
  gpu.resize(count);
  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, count * sizeof(int), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));
  PMPP_CUDA_CHECK(cudaFree(device_count));
  return gpu;
}

std::vector<int> run_stable_compaction(const std::vector<int> &input) {
  const int n = static_cast<int>(input.size());
  const int padded_n = next_power_of_two(n);
  std::vector<int> gpu(n, 0);

  int *device_input = nullptr;
  int *device_output = nullptr;
  int *device_count = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_count, sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemset(device_count, 0, sizeof(int)));

  compact_stable_scan_kernel<<<1, padded_n>>>(device_input, device_output, device_count, n,
                                              padded_n);
  PMPP_CUDA_KERNEL_CHECK();

  int count = 0;
  PMPP_CUDA_CHECK(cudaMemcpy(&count, device_count, sizeof(int), cudaMemcpyDeviceToHost));
  gpu.resize(count);
  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, count * sizeof(int), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));
  PMPP_CUDA_CHECK(cudaFree(device_count));
  return gpu;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int n = clamp_problem_size(options.size);
  const std::vector<int> input = make_input(n, options.seed);
  const std::vector<int> cpu = cpu_reference(input);
  std::vector<int> gpu_atomic = run_atomic_compaction(input);
  const std::vector<int> gpu_stable = run_stable_compaction(input);

  std::sort(gpu_atomic.begin(), gpu_atomic.end());
  std::vector<int> cpu_sorted = cpu;
  std::sort(cpu_sorted.begin(), cpu_sorted.end());

  pmpp::ValidationSummary atomic_summary = pmpp::compare_vectors(cpu_sorted, gpu_atomic);
  pmpp::ValidationSummary stable_summary = pmpp::compare_vectors(cpu, gpu_stable);

  pmpp::ValidationSummary summary = stable_summary;
  summary.ok = atomic_summary.ok && stable_summary.ok;
  summary.mismatch_count = atomic_summary.mismatch_count + stable_summary.mismatch_count;
  summary.max_abs_error = std::max(atomic_summary.max_abs_error, stable_summary.max_abs_error);
  summary.notes =
      "Validated both an unordered atomic baseline and a stable single-block scan-based compactor.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int n = clamp_problem_size(options.size);
  const int padded_n = next_power_of_two(n);
  const std::vector<int> input = make_input(n, options.seed);

  int *device_input = nullptr;
  int *device_output = nullptr;
  int *device_count = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_count, sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    PMPP_CUDA_CHECK(cudaMemset(device_count, 0, sizeof(int)));
    compact_stable_scan_kernel<<<1, padded_n>>>(device_input, device_output, device_count, n,
                                                padded_n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(static_cast<std::size_t>(n) * sizeof(int) * 2,
                                              stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);
  stats.problem_label = "Input elements";
  stats.problem_size = static_cast<std::size_t>(n);

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));
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
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters,
                                 "Elements/sec");
  }

  return EXIT_SUCCESS;
}
