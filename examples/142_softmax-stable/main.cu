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

constexpr const char *kExampleName = "142_softmax-stable";
constexpr int kClasses = 128;

__global__ void softmax_stable_kernel(const float *logits, float *probabilities, int classes) {
  // One block handles one row of logits. The block first finds the row max for
  // numerical stability, then exponentiates and reduces the denominator.
  extern __shared__ float shared[];
  float *max_shared = shared;
  float *sum_shared = shared + blockDim.x;

  int row = blockIdx.x;
  int tid = threadIdx.x;
  int index = row * classes + tid;

  float value = -1.0e30f;
  if (tid < classes)
    value = logits[index];
  max_shared[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      max_shared[tid] = fmaxf(max_shared[tid], max_shared[tid + stride]);
    __syncthreads();
  }

  float row_max = max_shared[0];
  float exp_value = 0.0f;
  if (tid < classes)
    exp_value = expf(value - row_max);
  sum_shared[tid] = exp_value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      sum_shared[tid] += sum_shared[tid + stride];
    __syncthreads();
  }

  float denominator = sum_shared[0];
  if (tid < classes)
    probabilities[index] = exp_value / denominator;
}

std::vector<float> cpu_reference(const std::vector<float> &logits, int rows, int classes) {
  std::vector<float> output(logits.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    float row_max = logits[row * classes];
    for (int col = 1; col < classes; ++col)
      row_max = std::max(row_max, logits[row * classes + col]);

    double sum = 0.0;
    for (int col = 0; col < classes; ++col) {
      float shifted = logits[row * classes + col] - row_max;
      output[row * classes + col] = std::exp(shifted);
      sum += output[row * classes + col];
    }
    for (int col = 0; col < classes; ++col)
      output[row * classes + col] = static_cast<float>(output[row * classes + col] / sum);
  }
  return output;
}

pmpp::ValidationSummary run_check(int rows) {
  std::vector<float> logits = pmpp::make_uniform_floats(rows * kClasses, 142u, -5.0f, 5.0f);
  std::vector<float> expected = cpu_reference(logits, rows, kClasses);
  std::vector<float> actual(expected.size(), 0.0f);

  float *d_logits = nullptr;
  float *d_probabilities = nullptr;
  std::size_t bytes = static_cast<std::size_t>(logits.size()) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_logits, bytes));
  CUDA_CHECK(cudaMalloc(&d_probabilities, bytes));
  CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), bytes, cudaMemcpyHostToDevice));

  softmax_stable_kernel<<<rows, kClasses, 2 * kClasses * sizeof(float)>>>(
      d_logits, d_probabilities, kClasses);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual.data(), d_probabilities, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_logits));
  CUDA_CHECK(cudaFree(d_probabilities));

  return pmpp::compare_vectors(expected, actual, 1.0e-5f);
}

void run_bench(int rows, int warmup, int iters) {
  std::vector<float> logits = pmpp::make_uniform_floats(rows * kClasses, 142u, -5.0f, 5.0f);

  float *d_logits = nullptr;
  float *d_probabilities = nullptr;
  std::size_t bytes = static_cast<std::size_t>(logits.size()) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_logits, bytes));
  CUDA_CHECK(cudaMalloc(&d_probabilities, bytes));
  CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), bytes, cudaMemcpyHostToDevice));

  auto stats = pmpp::run_benchmark_loop(warmup, iters, [&] {
    softmax_stable_kernel<<<rows, kClasses, 2 * kClasses * sizeof(float)>>>(
        d_logits, d_probabilities, kClasses);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  });
  stats.problem_label = "Elements";
  stats.problem_size = rows * kClasses;
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(2 * bytes, stats.avg_ms);
  stats.throughput =
      pmpp::elements_per_second(static_cast<std::size_t>(rows) * kClasses, stats.avg_ms);

  pmpp::print_benchmark_report(kExampleName, stats, warmup, iters, "Elements/s");

  CUDA_CHECK(cudaFree(d_logits));
  CUDA_CHECK(cudaFree(d_probabilities));
}

}  // namespace

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  int rows = std::max(1, options.size / kClasses);

  pmpp::ValidationSummary summary = run_check(rows);
  pmpp::print_validation_report(kExampleName, summary);
  if (!summary.ok)
    return EXIT_FAILURE;

  if (options.bench)
    run_bench(rows, options.warmup, options.iters);

  return EXIT_SUCCESS;
}
