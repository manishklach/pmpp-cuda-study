// Example 098: Neural Network Forward Pass
// Track: Graph and ML
// Difficulty: Advanced
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

inline void check_cuda(cudaError_t status, const char *file, int line) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << file << ":" << line
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

__global__ void dense_forward_kernel(const float *input, const float *weight, const float *bias,
                                     float *output, int in_dim, int out_dim, bool relu) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out_dim)
    return;
  float sum = bias[o];
  for (int i = 0; i < in_dim; ++i)
    sum += weight[o * in_dim + i] * input[i];
  output[o] = relu ? fmaxf(0.0f, sum) : sum;
}

int main() {
  const int in_dim = 4, hidden_dim = 3, out_dim = 2;
  std::vector<float> input = {1.0f, -2.0f, 0.5f, 3.0f};
  std::vector<float> w1 = {0.2f, -0.1f, 0.4f, 0.3f, -0.5f, 0.2f,
                           0.1f, 0.6f,  0.3f, 0.2f, -0.2f, 0.5f};
  std::vector<float> b1 = {0.1f, -0.2f, 0.05f};
  std::vector<float> w2 = {0.3f, -0.4f, 0.2f, -0.1f, 0.5f, 0.6f};
  std::vector<float> b2 = {0.0f, 0.2f};
  std::vector<float> cpu_hidden(hidden_dim, 0.0f), cpu_out(out_dim, 0.0f),
      gpu_hidden(hidden_dim, 0.0f), gpu_out(out_dim, 0.0f);
  for (int o = 0; o < hidden_dim; ++o) {
    float sum = b1[o];
    for (int i = 0; i < in_dim; ++i)
      sum += w1[o * in_dim + i] * input[i];
    cpu_hidden[o] = std::max(0.0f, sum);
  }
  for (int o = 0; o < out_dim; ++o) {
    float sum = b2[o];
    for (int i = 0; i < hidden_dim; ++i)
      sum += w2[o * hidden_dim + i] * cpu_hidden[i];
    cpu_out[o] = sum;
  }

  float *d_input = nullptr, *d_w1 = nullptr, *d_b1 = nullptr, *d_hidden = nullptr;
  float *d_w2 = nullptr, *d_b2 = nullptr, *d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, in_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_w1, w1.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b1, hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_hidden, hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_w2, w2.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b2, out_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, out_dim * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_input, input.data(), in_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_w1, w1.data(), w1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b1, b1.data(), hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_w2, w2.data(), w2.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b2, b2.data(), out_dim * sizeof(float), cudaMemcpyHostToDevice));
  dense_forward_kernel<<<1, 256>>>(d_input, d_w1, d_b1, d_hidden, in_dim, hidden_dim, true);
  dense_forward_kernel<<<1, 256>>>(d_hidden, d_w2, d_b2, d_out, hidden_dim, out_dim, false);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(
      cudaMemcpy(gpu_hidden.data(), d_hidden, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(gpu_out.data(), d_out, out_dim * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < hidden_dim; ++i)
    if (std::fabs(cpu_hidden[i] - gpu_hidden[i]) > 1.0e-5f)
      ok = false;
  for (int i = 0; i < out_dim; ++i)
    if (std::fabs(cpu_out[i] - gpu_out[i]) > 1.0e-5f)
      ok = false;
  std::cout << "Output:";
  for (float value : gpu_out)
    std::cout << " " << value;
  std::cout << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_w1));
  CHECK_CUDA(cudaFree(d_b1));
  CHECK_CUDA(cudaFree(d_hidden));
  CHECK_CUDA(cudaFree(d_w2));
  CHECK_CUDA(cudaFree(d_b2));
  CHECK_CUDA(cudaFree(d_out));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
