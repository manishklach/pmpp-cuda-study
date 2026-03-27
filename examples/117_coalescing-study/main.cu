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

constexpr const char *kExampleName = "117_coalescing-study";
constexpr int kStride = 32;

__global__ void coalesced_copy_kernel(const float *input, float *output, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    output[i] = input[i];
}

__global__ void strided_copy_kernel(const float *input, float *output, int logical_n, int stride) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int global_stride = blockDim.x * gridDim.x;
  for (int i = index; i < logical_n; i += global_stride)
    output[i] = input[i * stride];
}

std::vector<float> cpu_coalesced_reference(const std::vector<float> &input, int n) {
  return std::vector<float>(input.begin(), input.begin() + n);
}

std::vector<float> cpu_strided_reference(const std::vector<float> &input, int logical_n) {
  std::vector<float> output(logical_n, 0.0f);
  for (int i = 0; i < logical_n; ++i)
    output[i] = input[i * kStride];
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int logical_n = std::max(256, options.size);
  const int total_n = logical_n * kStride;
  const int block_size = 256;
  const int blocks = std::max(1, (logical_n + block_size - 1) / block_size);
  std::vector<float> input = pmpp::make_uniform_floats(total_n, options.seed, -1.0f, 1.0f);
  std::vector<float> coalesced_cpu = cpu_coalesced_reference(input, logical_n);
  std::vector<float> strided_cpu = cpu_strided_reference(input, logical_n);
  std::vector<float> coalesced_gpu(logical_n, 0.0f);
  std::vector<float> strided_gpu(logical_n, 0.0f);

  float *device_input = nullptr;
  float *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, total_n * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, logical_n * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), total_n * sizeof(float), cudaMemcpyHostToDevice));

  coalesced_copy_kernel<<<blocks, block_size>>>(device_input, device_output, logical_n);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(coalesced_gpu.data(), device_output, logical_n * sizeof(float), cudaMemcpyDeviceToHost));

  strided_copy_kernel<<<blocks, block_size>>>(device_input, device_output, logical_n, kStride);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(strided_gpu.data(), device_output, logical_n * sizeof(float), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_input));
  PMPP_CUDA_CHECK(cudaFree(device_output));

  pmpp::ValidationSummary coalesced_summary = pmpp::compare_vectors(coalesced_cpu, coalesced_gpu, 1.0e-6f);
  pmpp::ValidationSummary strided_summary = pmpp::compare_vectors(strided_cpu, strided_gpu, 1.0e-6f);
  pmpp::ValidationSummary summary = strided_summary;
  summary.ok = coalesced_summary.ok && strided_summary.ok;
  summary.mismatch_count = coalesced_summary.mismatch_count + strided_summary.mismatch_count;
  summary.max_abs_error = std::max(coalesced_summary.max_abs_error, strided_summary.max_abs_error);
  summary.notes = "Both kernels are correct, but only the coalesced kernel lets neighboring threads read neighboring values.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int logical_n = std::max(256, options.size);
  const int total_n = logical_n * kStride;
  const int block_size = 256;
  const int blocks = std::max(1, (logical_n + block_size - 1) / block_size);
  std::vector<float> input = pmpp::make_uniform_floats(total_n, options.seed, -1.0f, 1.0f);
  float *device_input = nullptr;
  float *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_input, total_n * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, logical_n * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_input, input.data(), total_n * sizeof(float), cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    coalesced_copy_kernel<<<blocks, block_size>>>(device_input, device_output, logical_n);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(static_cast<std::size_t>(logical_n) * sizeof(float) * 2, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(logical_n, stats.avg_ms);
  stats.problem_label = "Logical output elements";
  stats.problem_size = static_cast<std::size_t>(logical_n);

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
