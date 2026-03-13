#include <cuda_runtime.h>

#include <cmath>
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
constexpr const char *kExampleName = "022_l2-norm";
constexpr int kMaxThreads = 256;

__global__ void squared_sum_partials_kernel(const float *x, float *partials, int n) {
  __shared__ float scratch[kMaxThreads];
  int global = blockIdx.x * blockDim.x + threadIdx.x;
  int local = threadIdx.x;
  scratch[local] = global < n ? x[global] * x[global] : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local < stride)
      scratch[local] += scratch[local + stride];
    __syncthreads();
  }
  if (local == 0)
    partials[blockIdx.x] = scratch[0];
}

double run_gpu_sqsum(const std::vector<float> &x, int block_size) {
  int n = static_cast<int>(x.size());
  int blocks = (n + block_size - 1) / block_size;
  std::vector<float> partials(blocks, 0.0f);
  float *dx = nullptr, *dp = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&dp, blocks * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  squared_sum_partials_kernel<<<blocks, block_size>>>(dx, dp, n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(partials.data(), dp, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  PMPP_CUDA_CHECK(cudaFree(dx));
  PMPP_CUDA_CHECK(cudaFree(dp));
  return std::accumulate(partials.begin(), partials.end(), 0.0);
}
}

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  if (options.block_size > kMaxThreads)
    options.block_size = kMaxThreads;
  std::vector<float> x = pmpp::make_uniform_floats(options.size, options.seed, -4.0f, 4.0f);

  if (options.check) {
    double cpu_sq = 0.0;
    for (float value : x)
      cpu_sq += static_cast<double>(value) * value;
    double cpu = std::sqrt(cpu_sq);
    double gpu = std::sqrt(run_gpu_sqsum(x, options.block_size));
    pmpp::ValidationSummary summary = pmpp::compare_scalars(cpu, gpu, 1.0e-3);
    summary.notes = "L2 norm reuses the reduction structure after squaring each input.";
    pmpp::print_validation_report(kExampleName, summary);
    if (!summary.ok)
      return EXIT_FAILURE;
  }

  if (options.bench) {
    int blocks = (options.size + options.block_size - 1) / options.block_size;
    float *dx = nullptr, *dp = nullptr;
    PMPP_CUDA_CHECK(cudaMalloc(&dx, options.size * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMalloc(&dp, blocks * sizeof(float)));
    PMPP_CUDA_CHECK(cudaMemcpy(dx, x.data(), options.size * sizeof(float), cudaMemcpyHostToDevice));
    pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
      squared_sum_partials_kernel<<<blocks, options.block_size>>>(dx, dp, options.size);
      PMPP_CUDA_KERNEL_CHECK();
    });
    stats.bandwidth_gbps =
        pmpp::bandwidth_gbps((static_cast<std::size_t>(options.size) + blocks) * sizeof(float),
                             stats.avg_ms);
    stats.throughput = pmpp::elements_per_second(options.size, stats.avg_ms);
    if (!options.verify)
      std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters,
                                 "Elements/sec");
    PMPP_CUDA_CHECK(cudaFree(dx));
    PMPP_CUDA_CHECK(cudaFree(dp));
  }

  return EXIT_SUCCESS;
}
