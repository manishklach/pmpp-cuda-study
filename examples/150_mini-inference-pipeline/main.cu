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

constexpr const char *kExampleName = "150_mini-inference-pipeline";
constexpr int kBatch = 8;
constexpr int kInputDim = 64;
constexpr int kHiddenDim = 64;
constexpr int kOutputDim = 16;
constexpr int kTile = 16;

__global__ void linear_relu_kernel(const float *input, const float *weights,
                                   const float *bias, float *output, int rows,
                                   int input_dim, int output_dim) {
  // One thread computes one output activation. Threads in the same block walk
  // across the input dimension together, which keeps the mapping easy to read.
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= output_dim)
    return;

  float sum = bias[col];
  for (int k = 0; k < input_dim; ++k)
    sum += input[row * input_dim + k] * weights[k * output_dim + col];
  output[row * output_dim + col] = fmaxf(sum, 0.0f);
}

__global__ void linear_kernel(const float *input, const float *weights, const float *bias,
                              float *output, int rows, int input_dim, int output_dim) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= output_dim)
    return;

  float sum = bias[col];
  for (int k = 0; k < input_dim; ++k)
    sum += input[row * input_dim + k] * weights[k * output_dim + col];
  output[row * output_dim + col] = sum;
}

__global__ void rowwise_softmax_kernel(const float *logits, float *probabilities, int classes) {
  // One block owns one batch row of logits, first reducing the row max and then
  // the exponentiated denominator before writing final probabilities.
  extern __shared__ float shared[];
  float *max_shared = shared;
  float *sum_shared = shared + blockDim.x;

  int row = blockIdx.x;
  int tid = threadIdx.x;
  int index = row * classes + tid;

  float value = tid < classes ? logits[index] : -1.0e30f;
  max_shared[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      max_shared[tid] = fmaxf(max_shared[tid], max_shared[tid + stride]);
    __syncthreads();
  }

  float row_max = max_shared[0];
  float exp_value = tid < classes ? expf(value - row_max) : 0.0f;
  sum_shared[tid] = exp_value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      sum_shared[tid] += sum_shared[tid + stride];
    __syncthreads();
  }

  if (tid < classes)
    probabilities[index] = exp_value / sum_shared[0];
}

void cpu_linear_relu(const std::vector<float> &input, const std::vector<float> &weights,
                     const std::vector<float> &bias, std::vector<float> &output, int rows,
                     int input_dim, int output_dim) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < output_dim; ++col) {
      float sum = bias[col];
      for (int k = 0; k < input_dim; ++k)
        sum += input[row * input_dim + k] * weights[k * output_dim + col];
      output[row * output_dim + col] = std::max(sum, 0.0f);
    }
  }
}

void cpu_linear(const std::vector<float> &input, const std::vector<float> &weights,
                const std::vector<float> &bias, std::vector<float> &output, int rows,
                int input_dim, int output_dim) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < output_dim; ++col) {
      float sum = bias[col];
      for (int k = 0; k < input_dim; ++k)
        sum += input[row * input_dim + k] * weights[k * output_dim + col];
      output[row * output_dim + col] = sum;
    }
  }
}

void cpu_softmax(std::vector<float> &logits, int rows, int classes) {
  for (int row = 0; row < rows; ++row) {
    float row_max = logits[row * classes];
    for (int col = 1; col < classes; ++col)
      row_max = std::max(row_max, logits[row * classes + col]);

    double sum = 0.0;
    for (int col = 0; col < classes; ++col) {
      logits[row * classes + col] = std::exp(logits[row * classes + col] - row_max);
      sum += logits[row * classes + col];
    }
    for (int col = 0; col < classes; ++col)
      logits[row * classes + col] = static_cast<float>(logits[row * classes + col] / sum);
  }
}

std::vector<float> cpu_reference(const std::vector<float> &input, const std::vector<float> &w1,
                                 const std::vector<float> &b1, const std::vector<float> &w2,
                                 const std::vector<float> &b2) {
  std::vector<float> hidden(kBatch * kHiddenDim, 0.0f);
  std::vector<float> logits(kBatch * kOutputDim, 0.0f);
  cpu_linear_relu(input, w1, b1, hidden, kBatch, kInputDim, kHiddenDim);
  cpu_linear(hidden, w2, b2, logits, kBatch, kHiddenDim, kOutputDim);
  cpu_softmax(logits, kBatch, kOutputDim);
  return logits;
}

pmpp::ValidationSummary run_check() {
  std::vector<float> input = pmpp::make_uniform_floats(kBatch * kInputDim, 150u, -1.0f, 1.0f);
  std::vector<float> w1 = pmpp::make_uniform_floats(kInputDim * kHiddenDim, 151u, -0.4f, 0.4f);
  std::vector<float> b1 = pmpp::make_uniform_floats(kHiddenDim, 152u, -0.1f, 0.1f);
  std::vector<float> w2 = pmpp::make_uniform_floats(kHiddenDim * kOutputDim, 153u, -0.3f, 0.3f);
  std::vector<float> b2 = pmpp::make_uniform_floats(kOutputDim, 154u, -0.1f, 0.1f);
  std::vector<float> expected = cpu_reference(input, w1, b1, w2, b2);
  std::vector<float> actual(expected.size(), 0.0f);

  float *d_input = nullptr;
  float *d_w1 = nullptr;
  float *d_b1 = nullptr;
  float *d_hidden = nullptr;
  float *d_w2 = nullptr;
  float *d_b2 = nullptr;
  float *d_logits = nullptr;
  float *d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_w1, w1.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b1, b1.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hidden, static_cast<std::size_t>(kBatch) * kHiddenDim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_w2, w2.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b2, b2.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_logits, static_cast<std::size_t>(kBatch) * kOutputDim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, static_cast<std::size_t>(kBatch) * kOutputDim * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w1, w1.data(), w1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w2, w2.data(), w2.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(kTile, kTile);
  dim3 grid1((kHiddenDim + block.x - 1) / block.x, (kBatch + block.y - 1) / block.y);
  dim3 grid2((kOutputDim + block.x - 1) / block.x, (kBatch + block.y - 1) / block.y);
  linear_relu_kernel<<<grid1, block>>>(d_input, d_w1, d_b1, d_hidden, kBatch, kInputDim,
                                       kHiddenDim);
  CUDA_CHECK(cudaGetLastError());
  linear_kernel<<<grid2, block>>>(d_hidden, d_w2, d_b2, d_logits, kBatch, kHiddenDim,
                                  kOutputDim);
  CUDA_CHECK(cudaGetLastError());
  rowwise_softmax_kernel<<<kBatch, 32, 64 * sizeof(float)>>>(d_logits, d_output, kOutputDim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual.data(), d_output,
                        static_cast<std::size_t>(kBatch) * kOutputDim * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_w1));
  CUDA_CHECK(cudaFree(d_b1));
  CUDA_CHECK(cudaFree(d_hidden));
  CUDA_CHECK(cudaFree(d_w2));
  CUDA_CHECK(cudaFree(d_b2));
  CUDA_CHECK(cudaFree(d_logits));
  CUDA_CHECK(cudaFree(d_output));

  return pmpp::compare_vectors(expected, actual, 1.0e-4f);
}

void run_bench(int warmup, int iters) {
  std::vector<float> input = pmpp::make_uniform_floats(kBatch * kInputDim, 150u, -1.0f, 1.0f);
  std::vector<float> w1 = pmpp::make_uniform_floats(kInputDim * kHiddenDim, 151u, -0.4f, 0.4f);
  std::vector<float> b1 = pmpp::make_uniform_floats(kHiddenDim, 152u, -0.1f, 0.1f);
  std::vector<float> w2 = pmpp::make_uniform_floats(kHiddenDim * kOutputDim, 153u, -0.3f, 0.3f);
  std::vector<float> b2 = pmpp::make_uniform_floats(kOutputDim, 154u, -0.1f, 0.1f);

  float *d_input = nullptr;
  float *d_w1 = nullptr;
  float *d_b1 = nullptr;
  float *d_hidden = nullptr;
  float *d_w2 = nullptr;
  float *d_b2 = nullptr;
  float *d_logits = nullptr;
  float *d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_w1, w1.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b1, b1.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hidden, static_cast<std::size_t>(kBatch) * kHiddenDim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_w2, w2.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b2, b2.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_logits, static_cast<std::size_t>(kBatch) * kOutputDim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, static_cast<std::size_t>(kBatch) * kOutputDim * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w1, w1.data(), w1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w2, w2.data(), w2.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(kTile, kTile);
  dim3 grid1((kHiddenDim + block.x - 1) / block.x, (kBatch + block.y - 1) / block.y);
  dim3 grid2((kOutputDim + block.x - 1) / block.x, (kBatch + block.y - 1) / block.y);

  auto stats = pmpp::run_benchmark_loop(warmup, iters, [&] {
    linear_relu_kernel<<<grid1, block>>>(d_input, d_w1, d_b1, d_hidden, kBatch, kInputDim,
                                         kHiddenDim);
    CUDA_CHECK(cudaGetLastError());
    linear_kernel<<<grid2, block>>>(d_hidden, d_w2, d_b2, d_logits, kBatch, kHiddenDim,
                                    kOutputDim);
    CUDA_CHECK(cudaGetLastError());
    rowwise_softmax_kernel<<<kBatch, 32, 64 * sizeof(float)>>>(d_logits, d_output, kOutputDim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  });
  stats.problem_label = "Batch";
  stats.problem_size = kBatch;
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(
      (input.size() + w1.size() + b1.size() + w2.size() + b2.size() + kBatch * kHiddenDim +
       2 * kBatch * kOutputDim) *
          sizeof(float),
      stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(kBatch, stats.avg_ms);

  pmpp::print_benchmark_report(kExampleName, stats, warmup, iters, "Batches/s");

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_w1));
  CUDA_CHECK(cudaFree(d_b1));
  CUDA_CHECK(cudaFree(d_hidden));
  CUDA_CHECK(cudaFree(d_w2));
  CUDA_CHECK(cudaFree(d_b2));
  CUDA_CHECK(cudaFree(d_logits));
  CUDA_CHECK(cudaFree(d_output));
}

}  // namespace

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);

  pmpp::ValidationSummary summary = run_check();
  pmpp::print_validation_report(kExampleName, summary);
  if (!summary.ok)
    return EXIT_FAILURE;

  if (options.bench)
    run_bench(options.warmup, options.iters);

  return EXIT_SUCCESS;
}
