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

constexpr const char *kExampleName = "102_segmented-scan";
constexpr int kMaxElements = 256;

int clamp_problem_size(int requested) {
  return std::max(32, std::min(requested, kMaxElements));
}

__global__ void segmented_scan_kernel(const int *input, const int *head_flags, int *output, int n) {
  __shared__ int values[kMaxElements];
  __shared__ int heads[kMaxElements];
  int tid = threadIdx.x;

  values[tid] = tid < n ? input[tid] : 0;
  heads[tid] = tid < n ? head_flags[tid] : 1;
  __syncthreads();

  // This Hillis-Steele style scan propagates head flags alongside values so additions stop at the
  // start of each segment. It is synchronization-heavy, but the segment-reset logic is explicit.
  for (int offset = 1; offset < n; offset <<= 1) {
    int current_value = values[tid];
    int current_head = heads[tid];
    int addend = 0;
    int next_head = current_head;

    if (tid < n && tid >= offset) {
      int left_value = values[tid - offset];
      int left_head = heads[tid - offset];
      if (current_head == 0)
        addend = left_value;
      next_head = current_head | left_head;
    }
    __syncthreads();

    if (tid < n) {
      values[tid] = current_value + addend;
      heads[tid] = next_head;
    }
    __syncthreads();
  }

  if (tid < n)
    output[tid] = values[tid];
}

std::vector<int> make_input(int n) {
  std::vector<int> input(n, 0);
  for (int i = 0; i < n; ++i)
    input[i] = (i % 7) + 1;
  return input;
}

std::vector<int> make_head_flags(int n) {
  std::vector<int> heads(n, 0);
  for (int i = 0; i < n; ++i)
    heads[i] = (i % 16 == 0) ? 1 : 0;
  if (!heads.empty())
    heads[0] = 1;
  return heads;
}

std::vector<int> cpu_reference(const std::vector<int> &input, const std::vector<int> &heads) {
  std::vector<int> output(input.size(), 0);
  int running = 0;
  for (std::size_t i = 0; i < input.size(); ++i) {
    if (heads[i])
      running = 0;
    running += input[i];
    output[i] = running;
  }
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int n = clamp_problem_size(options.size);
  std::vector<int> input = make_input(n);
  std::vector<int> heads = make_head_flags(n);
  std::vector<int> cpu = cpu_reference(input, heads);
  std::vector<int> gpu(n, 0);

  int *device_input = nullptr;
  int *device_heads = nullptr;
  int *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_heads, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_heads, heads.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  segmented_scan_kernel<<<1, kMaxElements>>>(device_input, device_heads, device_output, n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, n * sizeof(int), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_heads));
  PMPP_CUDA_CHECK(cudaFree(device_output));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu);
  summary.notes = "This segmented scan keeps the head-flag propagation explicit so segment resets are easy to inspect.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int n = clamp_problem_size(options.size);
  std::vector<int> input = make_input(n);
  std::vector<int> heads = make_head_flags(n);

  int *device_input = nullptr;
  int *device_heads = nullptr;
  int *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_heads, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_heads, heads.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    segmented_scan_kernel<<<1, kMaxElements>>>(device_input, device_heads, device_output, n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(static_cast<std::size_t>(n) * sizeof(int) * 3, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);
  stats.problem_label = "Scanned elements";
  stats.problem_size = static_cast<std::size_t>(n);

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_heads));
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
