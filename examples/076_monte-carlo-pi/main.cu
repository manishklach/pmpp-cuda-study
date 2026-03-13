#include <cuda_runtime.h>

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
constexpr const char *kExampleName = "076_monte-carlo-pi";
__device__ unsigned int lcg_next(unsigned int &state) {
  state = 1664525u * state + 1013904223u;
  return state;
}
__device__ float uniform01(unsigned int &state) {
  return (lcg_next(state) & 0x00FFFFFF) / static_cast<float>(0x01000000);
}
__global__ void monte_carlo_pi_kernel(int trials_per_thread, int *hits) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int state = 1234567u + 747796405u * static_cast<unsigned int>(idx);
  int local_hits = 0;
  for (int i = 0; i < trials_per_thread; ++i) {
    float x = uniform01(state);
    float y = uniform01(state);
    if (x * x + y * y <= 1.0f) ++local_hits;
  }
  hits[idx] = local_hits;
}
}

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  int threads = 256;
  int blocks = std::max(1, options.size / threads);
  int total_threads = threads * blocks;
  int trials_per_thread = 1024;
  std::vector<int> hits(total_threads, 0);

  auto run_once = [&]() {
    int *d_hits = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&d_hits, total_threads * sizeof(int)));
    monte_carlo_pi_kernel<<<blocks, threads>>>(trials_per_thread, d_hits);
    PMPP_CUDA_KERNEL_CHECK();
    PMPP_CUDA_CHECK(cudaMemcpy(hits.data(), d_hits, total_threads * sizeof(int), cudaMemcpyDeviceToHost));
    PMPP_CUDA_CHECK(cudaFree(d_hits));
    long long total_hits = 0;
    for (int value : hits) total_hits += value;
    long long total_trials = static_cast<long long>(total_threads) * trials_per_thread;
    return 4.0 * static_cast<double>(total_hits) / static_cast<double>(total_trials);
  };

  if (options.check) {
    double estimate = run_once();
    pmpp::ValidationSummary summary = pmpp::compare_scalars(3.141592653589793, estimate, 0.05);
    summary.notes = "Monte Carlo validation is tolerance-based because the estimator is stochastic.";
    pmpp::print_validation_report(kExampleName, summary);
    if (!summary.ok) return EXIT_FAILURE;
  }
  if (options.bench) {
    int *d_hits = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&d_hits, total_threads * sizeof(int)));
    pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
      monte_carlo_pi_kernel<<<blocks, threads>>>(trials_per_thread, d_hits);
      PMPP_CUDA_KERNEL_CHECK();
    });
    stats.bandwidth_gbps = pmpp::bandwidth_gbps(total_threads * sizeof(int), stats.avg_ms);
    stats.throughput = pmpp::elements_per_second(static_cast<std::size_t>(total_threads) * trials_per_thread, stats.avg_ms);
    if (!options.verify) std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Samples/sec");
    PMPP_CUDA_CHECK(cudaFree(d_hits));
  }
  return EXIT_SUCCESS;
}
