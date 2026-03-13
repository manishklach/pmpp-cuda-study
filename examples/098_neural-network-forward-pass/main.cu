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

constexpr const char *kExampleName = "098_neural-network-forward-pass";

__global__ void dense_forward_kernel(const float *input, const float *weight, const float *bias,
                                     float *output, int in_dim, int out_dim, bool relu) {
  int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_idx >= out_dim)
    return;

  float sum = bias[out_idx];
  for (int i = 0; i < in_dim; ++i)
    sum += weight[out_idx * in_dim + i] * input[i];
  output[out_idx] = relu ? fmaxf(0.0f, sum) : sum;
}

struct MlpProblem {
  int in_dim = 0;
  int hidden_dim = 0;
  int out_dim = 0;
  std::vector<float> input;
  std::vector<float> w1;
  std::vector<float> b1;
  std::vector<float> w2;
  std::vector<float> b2;
};

MlpProblem make_problem(int size, unsigned int seed) {
  MlpProblem p;
  p.in_dim = std::max(4, size);
  p.hidden_dim = std::max(4, size / 2);
  p.out_dim = std::max(2, size / 4);
  p.input = pmpp::make_uniform_floats(p.in_dim, seed, -1.0f, 1.0f);
  p.w1 = pmpp::make_uniform_floats(p.hidden_dim * p.in_dim, seed + 1, -0.5f, 0.5f);
  p.b1 = pmpp::make_uniform_floats(p.hidden_dim, seed + 2, -0.1f, 0.1f);
  p.w2 = pmpp::make_uniform_floats(p.out_dim * p.hidden_dim, seed + 3, -0.5f, 0.5f);
  p.b2 = pmpp::make_uniform_floats(p.out_dim, seed + 4, -0.1f, 0.1f);
  return p;
}

std::vector<float> cpu_layer(const std::vector<float> &input, const std::vector<float> &weight,
                             const std::vector<float> &bias, int in_dim, int out_dim, bool relu) {
  std::vector<float> output(out_dim, 0.0f);
  for (int o = 0; o < out_dim; ++o) {
    float sum = bias[o];
    for (int i = 0; i < in_dim; ++i)
      sum += weight[o * in_dim + i] * input[i];
    output[o] = relu ? std::max(0.0f, sum) : sum;
  }
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  MlpProblem p = make_problem(options.size, options.seed);
  std::vector<float> cpu_hidden = cpu_layer(p.input, p.w1, p.b1, p.in_dim, p.hidden_dim, true);
  std::vector<float> cpu_out = cpu_layer(cpu_hidden, p.w2, p.b2, p.hidden_dim, p.out_dim, false);
  std::vector<float> gpu_hidden(p.hidden_dim, 0.0f);
  std::vector<float> gpu_out(p.out_dim, 0.0f);

  float *d_input = nullptr, *d_w1 = nullptr, *d_b1 = nullptr, *d_hidden = nullptr;
  float *d_w2 = nullptr, *d_b2 = nullptr, *d_out = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&d_input, p.in_dim * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_w1, p.w1.size() * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_b1, p.hidden_dim * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_hidden, p.hidden_dim * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_w2, p.w2.size() * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_b2, p.out_dim * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_out, p.out_dim * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(d_input, p.input.data(), p.in_dim * sizeof(float), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(d_w1, p.w1.data(), p.w1.size() * sizeof(float), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(d_b1, p.b1.data(), p.hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(d_w2, p.w2.data(), p.w2.size() * sizeof(float), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(d_b2, p.b2.data(), p.out_dim * sizeof(float), cudaMemcpyHostToDevice));

  dense_forward_kernel<<<(p.hidden_dim + 255) / 256, 256>>>(d_input, d_w1, d_b1, d_hidden, p.in_dim,
                                                            p.hidden_dim, true);
  dense_forward_kernel<<<(p.out_dim + 255) / 256, 256>>>(d_hidden, d_w2, d_b2, d_out, p.hidden_dim,
                                                         p.out_dim, false);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(
      cudaMemcpy(gpu_hidden.data(), d_hidden, p.hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
  PMPP_CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_out, p.out_dim * sizeof(float), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(d_input));
  PMPP_CUDA_CHECK(cudaFree(d_w1));
  PMPP_CUDA_CHECK(cudaFree(d_b1));
  PMPP_CUDA_CHECK(cudaFree(d_hidden));
  PMPP_CUDA_CHECK(cudaFree(d_w2));
  PMPP_CUDA_CHECK(cudaFree(d_b2));
  PMPP_CUDA_CHECK(cudaFree(d_out));

  pmpp::ValidationSummary hidden = pmpp::compare_vectors(cpu_hidden, gpu_hidden, 1.0e-5f);
  pmpp::ValidationSummary output = pmpp::compare_vectors(cpu_out, gpu_out, 1.0e-5f);
  pmpp::ValidationSummary summary{};
  summary.ok = hidden.ok && output.ok;
  summary.max_abs_error = std::max(hidden.max_abs_error, output.max_abs_error);
  summary.mismatch_count = hidden.mismatch_count + output.mismatch_count;
  summary.notes = "This MLP forward pass uses two dense layers with ReLU in the hidden layer.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  MlpProblem p = make_problem(std::max(32, options.size), options.seed);

  float *d_input = nullptr, *d_w1 = nullptr, *d_b1 = nullptr, *d_hidden = nullptr;
  float *d_w2 = nullptr, *d_b2 = nullptr, *d_out = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&d_input, p.in_dim * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_w1, p.w1.size() * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_b1, p.hidden_dim * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_hidden, p.hidden_dim * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_w2, p.w2.size() * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_b2, p.out_dim * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&d_out, p.out_dim * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(d_input, p.input.data(), p.in_dim * sizeof(float), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(d_w1, p.w1.data(), p.w1.size() * sizeof(float), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(d_b1, p.b1.data(), p.hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(d_w2, p.w2.data(), p.w2.size() * sizeof(float), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(d_b2, p.b2.data(), p.out_dim * sizeof(float), cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    dense_forward_kernel<<<(p.hidden_dim + 255) / 256, 256>>>(d_input, d_w1, d_b1, d_hidden, p.in_dim,
                                                              p.hidden_dim, true);
    dense_forward_kernel<<<(p.out_dim + 255) / 256, 256>>>(d_hidden, d_w2, d_b2, d_out, p.hidden_dim,
                                                           p.out_dim, false);
    PMPP_CUDA_KERNEL_CHECK();
  });
  std::size_t bytes = (p.input.size() + p.w1.size() + p.b1.size() + p.hidden_dim + p.w2.size() +
                       p.b2.size() + p.out_dim) * sizeof(float);
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(bytes, stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(p.out_dim, stats.avg_ms);

  PMPP_CUDA_CHECK(cudaFree(d_input));
  PMPP_CUDA_CHECK(cudaFree(d_w1));
  PMPP_CUDA_CHECK(cudaFree(d_b1));
  PMPP_CUDA_CHECK(cudaFree(d_hidden));
  PMPP_CUDA_CHECK(cudaFree(d_w2));
  PMPP_CUDA_CHECK(cudaFree(d_b2));
  PMPP_CUDA_CHECK(cudaFree(d_out));
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
                                 "Output neurons/sec");
  }

  return EXIT_SUCCESS;
}
