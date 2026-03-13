#include <cuda_runtime.h>

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

constexpr const char *kExampleName = "028_histogram-global-atomics";
constexpr int kNumBins = 16;

__global__ void histogram_global_kernel(const unsigned int *input, unsigned int *bins, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    atomicAdd(&bins[input[idx] % kNumBins], 1u);
}

std::vector<unsigned int> make_input(int n, unsigned int seed) {
  std::vector<int> ints = pmpp::make_uniform_ints(n, seed, 0, kNumBins - 1);
  std::vector<unsigned int> output(n, 0);
  for (int i = 0; i < n; ++i)
    output[i] = static_cast<unsigned int>(ints[i]);
  return output;
}

std::vector<unsigned int> cpu_reference(const std::vector<unsigned int> &input) {
  std::vector<unsigned int> bins(kNumBins, 0);
  for (unsigned int value : input)
    ++bins[value % kNumBins];
  return bins;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int n = options.size;
  std::vector<unsigned int> input = make_input(n, options.seed);
  std::vector<unsigned int> cpu = cpu_reference(input);
  std::vector<unsigned int> gpu(kNumBins, 0);

  unsigned int *device_input = nullptr;
  unsigned int *device_bins = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(unsigned int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_bins, kNumBins * sizeof(unsigned int)));
  PMPP_CUDA_CHECK(
      cudaMemcpy(device_input, input.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemset(device_bins, 0, kNumBins * sizeof(unsigned int)));

  const int threads = options.block_size;
  const int blocks = (n + threads - 1) / threads;
  histogram_global_kernel<<<blocks, threads>>>(device_input, device_bins, n);
  PMPP_CUDA_KERNEL_CHECK();

  PMPP_CUDA_CHECK(
      cudaMemcpy(gpu.data(), device_bins, kNumBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_bins));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu);
  summary.notes = "This is the direct global-atomics baseline for the histogram study path.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int n = options.size;
  std::vector<unsigned int> input = make_input(n, options.seed);
  unsigned int *device_input = nullptr;
  unsigned int *device_bins = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(unsigned int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_bins, kNumBins * sizeof(unsigned int)));
  PMPP_CUDA_CHECK(
      cudaMemcpy(device_input, input.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));

  const int threads = options.block_size;
  const int blocks = (n + threads - 1) / threads;
  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    PMPP_CUDA_CHECK(cudaMemset(device_bins, 0, kNumBins * sizeof(unsigned int)));
    histogram_global_kernel<<<blocks, threads>>>(device_input, device_bins, n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps =
      pmpp::bandwidth_gbps(static_cast<std::size_t>(n + kNumBins) * sizeof(unsigned int),
                           stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_bins));
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
