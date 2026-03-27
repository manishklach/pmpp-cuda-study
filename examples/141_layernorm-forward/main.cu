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
#include "pmpp/random_inputs.cuh"
#include "pmpp/report.cuh"

namespace {

constexpr const char *kExampleName = "141_layernorm-forward";
constexpr int kHidden = 256;
constexpr float kEpsilon = 1.0e-5f;

__global__ void layernorm_forward_kernel(const float *input, const float *gamma,
                                         const float *beta, float *output,
                                         int hidden_size) {
  // One block owns one row. Threads first accumulate row statistics, then
  // normalize with shared mean and variance before writing their output lane.
  extern __shared__ float shared[];
  float *sum_shared = shared;
  float *sq_sum_shared = shared + blockDim.x;

  int row = blockIdx.x;
  int tid = threadIdx.x;
  int index = row * hidden_size + tid;

  float value = 0.0f;
  if (tid < hidden_size)
    value = input[index];

  sum_shared[tid] = value;
  sq_sum_shared[tid] = value * value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum_shared[tid] += sum_shared[tid + stride];
      sq_sum_shared[tid] += sq_sum_shared[tid + stride];
    }
    __syncthreads();
  }

  float mean = sum_shared[0] / static_cast<float>(hidden_size);
  float variance = sq_sum_shared[0] / static_cast<float>(hidden_size) - mean * mean;
  float inv_std = rsqrtf(variance + kEpsilon);

  if (tid < hidden_size) {
    float normalized = (value - mean) * inv_std;
    output[index] = normalized * gamma[tid] + beta[tid];
  }
}

std::vector<float> cpu_reference(const std::vector<float> &input,
                                 const std::vector<float> &gamma,
                                 const std::vector<float> &beta, int rows,
                                 int hidden_size) {
  std::vector<float> output(input.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    double sum = 0.0;
    double sq_sum = 0.0;
    for (int col = 0; col < hidden_size; ++col) {
      float value = input[row * hidden_size + col];
      sum += value;
      sq_sum += static_cast<double>(value) * value;
    }

    double mean = sum / static_cast<double>(hidden_size);
    double variance = sq_sum / static_cast<double>(hidden_size) - mean * mean;
    double inv_std = 1.0 / std::sqrt(variance + kEpsilon);

    for (int col = 0; col < hidden_size; ++col) {
      float value = input[row * hidden_size + col];
      float normalized = static_cast<float>((value - mean) * inv_std);
      output[row * hidden_size + col] = normalized * gamma[col] + beta[col];
    }
  }
  return output;
}

pmpp::ValidationSummary run_check(int rows) {
  std::vector<float> input = pmpp::make_uniform_floats(rows * kHidden, 141u, -2.0f, 2.0f);
  std::vector<float> gamma = pmpp::make_uniform_floats(kHidden, 142u, 0.5f, 1.5f);
  std::vector<float> beta = pmpp::make_uniform_floats(kHidden, 143u, -0.25f, 0.25f);
  std::vector<float> expected = cpu_reference(input, gamma, beta, rows, kHidden);
  std::vector<float> actual(expected.size(), 0.0f);

  float *d_input = nullptr;
  float *d_gamma = nullptr;
  float *d_beta = nullptr;
  float *d_output = nullptr;
  std::size_t bytes = static_cast<std::size_t>(input.size()) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_gamma, kHidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_beta, kHidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), kHidden * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), kHidden * sizeof(float), cudaMemcpyHostToDevice));

  layernorm_forward_kernel<<<rows, kHidden, 2 * kHidden * sizeof(float)>>>(
      d_input, d_gamma, d_beta, d_output, kHidden);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual.data(), d_output, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_beta));
  CUDA_CHECK(cudaFree(d_output));

  return pmpp::compare_vectors(expected, actual, 1.0e-4f);
}

void run_bench(int rows, int warmup, int iters) {
  std::vector<float> input = pmpp::make_uniform_floats(rows * kHidden, 141u, -2.0f, 2.0f);
  std::vector<float> gamma = pmpp::make_uniform_floats(kHidden, 142u, 0.5f, 1.5f);
  std::vector<float> beta = pmpp::make_uniform_floats(kHidden, 143u, -0.25f, 0.25f);

  float *d_input = nullptr;
  float *d_gamma = nullptr;
  float *d_beta = nullptr;
  float *d_output = nullptr;
  std::size_t bytes = static_cast<std::size_t>(input.size()) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_gamma, kHidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_beta, kHidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), kHidden * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), kHidden * sizeof(float), cudaMemcpyHostToDevice));

  auto stats = pmpp::run_benchmark_loop(warmup, iters, [&] {
    layernorm_forward_kernel<<<rows, kHidden, 2 * kHidden * sizeof(float)>>>(
        d_input, d_gamma, d_beta, d_output, kHidden);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  });
  stats.problem_label = "Elements";
  stats.problem_size = rows * kHidden;
  stats.bandwidth_gbps =
      pmpp::bandwidth_gbps(3 * bytes + 2 * kHidden * sizeof(float), stats.avg_ms);
  stats.throughput =
      pmpp::elements_per_second(static_cast<std::size_t>(rows) * kHidden, stats.avg_ms);

  pmpp::print_benchmark_report(kExampleName, stats, warmup, iters, "Elements/s");

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_beta));
  CUDA_CHECK(cudaFree(d_output));
}

}  // namespace

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  int rows = std::max(1, options.size / kHidden);

  pmpp::ValidationSummary summary = run_check(rows);
  pmpp::print_validation_report(kExampleName, summary);
  if (!summary.ok)
    return EXIT_FAILURE;

  if (options.bench)
    run_bench(rows, options.warmup, options.iters);

  return EXIT_SUCCESS;
}
