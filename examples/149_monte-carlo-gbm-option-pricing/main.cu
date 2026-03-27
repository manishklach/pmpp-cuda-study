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

constexpr const char *kExampleName = "149_monte-carlo-gbm-option-pricing";
constexpr float kInitialPrice = 100.0f;
constexpr float kStrike = 105.0f;
constexpr float kRiskFreeRate = 0.03f;
constexpr float kVolatility = 0.2f;
constexpr float kTimeToMaturity = 1.0f;
constexpr int kBlockSize = 256;

__device__ unsigned int lcg_step(unsigned int &state) {
  state = state * 1664525u + 1013904223u;
  return state;
}

__device__ float uniform01(unsigned int &state) {
  return (static_cast<float>(lcg_step(state) & 0x00ffffffu) + 1.0f) / 16777217.0f;
}

__device__ float normal_box_muller(unsigned int &state) {
  float u1 = uniform01(state);
  float u2 = uniform01(state);
  float radius = sqrtf(-2.0f * logf(u1));
  float angle = 6.28318530718f * u2;
  return radius * cosf(angle);
}

__global__ void monte_carlo_payoff_kernel(float *payoffs, int paths, unsigned int seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= paths)
    return;

  // Each thread owns one path. The work is embarrassingly parallel, so the main
  // study value is deterministic random generation and the final reduction.
  unsigned int state = seed + static_cast<unsigned int>(idx) * 747796405u;
  float z = normal_box_muller(state);
  float drift = (kRiskFreeRate - 0.5f * kVolatility * kVolatility) * kTimeToMaturity;
  float diffusion = kVolatility * sqrtf(kTimeToMaturity) * z;
  float terminal_price = kInitialPrice * expf(drift + diffusion);
  payoffs[idx] = fmaxf(terminal_price - kStrike, 0.0f);
}

__global__ void reduce_sum_kernel(const float *input, float *block_sums, int size) {
  __shared__ float shared[kBlockSize];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  shared[tid] = idx < size ? input[idx] : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      shared[tid] += shared[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    block_sums[blockIdx.x] = shared[0];
}

float cpu_reference(int paths, unsigned int seed) {
  auto lcg_host = [](unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
  };
  auto uniform_host = [&](unsigned int &state) {
    return (static_cast<float>(lcg_host(state) & 0x00ffffffu) + 1.0f) / 16777217.0f;
  };

  double sum = 0.0;
  for (int idx = 0; idx < paths; ++idx) {
    unsigned int state = seed + static_cast<unsigned int>(idx) * 747796405u;
    float u1 = uniform_host(state);
    float u2 = uniform_host(state);
    float radius = std::sqrt(-2.0f * std::log(u1));
    float angle = 6.28318530718f * u2;
    float z = radius * std::cos(angle);
    float drift = (kRiskFreeRate - 0.5f * kVolatility * kVolatility) * kTimeToMaturity;
    float diffusion = kVolatility * std::sqrt(kTimeToMaturity) * z;
    float terminal_price = kInitialPrice * std::exp(drift + diffusion);
    sum += std::max(terminal_price - kStrike, 0.0f);
  }

  double discount = std::exp(-kRiskFreeRate * kTimeToMaturity);
  return static_cast<float>(discount * sum / static_cast<double>(paths));
}

float gpu_price(int paths, unsigned int seed) {
  int blocks = (paths + kBlockSize - 1) / kBlockSize;

  float *d_payoffs = nullptr;
  float *d_block_sums = nullptr;
  std::vector<float> block_sums(blocks, 0.0f);
  CUDA_CHECK(cudaMalloc(&d_payoffs, static_cast<std::size_t>(paths) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_block_sums, static_cast<std::size_t>(blocks) * sizeof(float)));

  monte_carlo_payoff_kernel<<<blocks, kBlockSize>>>(d_payoffs, paths, seed);
  CUDA_CHECK(cudaGetLastError());
  reduce_sum_kernel<<<blocks, kBlockSize>>>(d_payoffs, d_block_sums, paths);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(block_sums.data(), d_block_sums,
                        static_cast<std::size_t>(blocks) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_payoffs));
  CUDA_CHECK(cudaFree(d_block_sums));

  double sum = 0.0;
  for (float value : block_sums)
    sum += value;
  double discount = std::exp(-kRiskFreeRate * kTimeToMaturity);
  return static_cast<float>(discount * sum / static_cast<double>(paths));
}

pmpp::ValidationSummary run_check(int paths, unsigned int seed) {
  float expected = cpu_reference(paths, seed);
  float actual = gpu_price(paths, seed);
  return pmpp::compare_scalars(expected, actual, 1.0e-3);
}

void run_bench(int paths, unsigned int seed, int warmup, int iters) {
  int blocks = (paths + kBlockSize - 1) / kBlockSize;
  float *d_payoffs = nullptr;
  float *d_block_sums = nullptr;
  CUDA_CHECK(cudaMalloc(&d_payoffs, static_cast<std::size_t>(paths) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_block_sums, static_cast<std::size_t>(blocks) * sizeof(float)));

  auto stats = pmpp::run_benchmark_loop(warmup, iters, [&] {
    monte_carlo_payoff_kernel<<<blocks, kBlockSize>>>(d_payoffs, paths, seed);
    CUDA_CHECK(cudaGetLastError());
    reduce_sum_kernel<<<blocks, kBlockSize>>>(d_payoffs, d_block_sums, paths);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  });
  stats.problem_label = "Paths";
  stats.problem_size = paths;
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(
      static_cast<std::size_t>(paths + blocks) * sizeof(float), stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(paths, stats.avg_ms);

  pmpp::print_benchmark_report(kExampleName, stats, warmup, iters, "Paths/s");

  CUDA_CHECK(cudaFree(d_payoffs));
  CUDA_CHECK(cudaFree(d_block_sums));
}

}  // namespace

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  int paths = std::max(1024, options.size);
  unsigned int seed = options.seed ? options.seed : 149u;

  pmpp::ValidationSummary summary = run_check(paths, seed);
  pmpp::print_validation_report(kExampleName, summary);
  if (!summary.ok)
    return EXIT_FAILURE;

  if (options.bench)
    run_bench(paths, seed, options.warmup, options.iters);

  return EXIT_SUCCESS;
}
