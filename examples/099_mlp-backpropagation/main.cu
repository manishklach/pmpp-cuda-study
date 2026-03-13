// Example 099: MLP Backpropagation
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

__global__ void output_grad_kernel(const float *hidden, const float *prediction,
                                   const float *target, float *grad_w, float *grad_b,
                                   int hidden_dim, int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = hidden_dim * out_dim;
  if (idx < total) {
    int o = idx / hidden_dim;
    int h = idx % hidden_dim;
    float delta = prediction[o] - target[o];
    grad_w[idx] = delta * hidden[h];
  }
  if (idx < out_dim)
    grad_b[idx] = prediction[idx] - target[idx];
}

int main() {
  const int hidden_dim = 3, out_dim = 2;
  std::vector<float> hidden = {1.2f, 0.0f, 0.7f};
  std::vector<float> prediction = {0.9f, -0.3f};
  std::vector<float> target = {1.0f, 0.2f};
  std::vector<float> cpu_w(hidden_dim * out_dim, 0.0f), cpu_b(out_dim, 0.0f);
  std::vector<float> gpu_w(cpu_w.size(), 0.0f), gpu_b(cpu_b.size(), 0.0f);
  for (int o = 0; o < out_dim; ++o) {
    float delta = prediction[o] - target[o];
    cpu_b[o] = delta;
    for (int h = 0; h < hidden_dim; ++h)
      cpu_w[o * hidden_dim + h] = delta * hidden[h];
  }

  float *d_hidden = nullptr, *d_prediction = nullptr, *d_target = nullptr, *d_grad_w = nullptr,
        *d_grad_b = nullptr;
  CHECK_CUDA(cudaMalloc(&d_hidden, hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_prediction, out_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_target, out_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_grad_w, gpu_w.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_grad_b, gpu_b.size() * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(d_hidden, hidden.data(), hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_prediction, prediction.data(), out_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_target, target.data(), out_dim * sizeof(float), cudaMemcpyHostToDevice));
  output_grad_kernel<<<1, 256>>>(d_hidden, d_prediction, d_target, d_grad_w, d_grad_b, hidden_dim,
                                 out_dim);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(
      cudaMemcpy(gpu_w.data(), d_grad_w, gpu_w.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(gpu_b.data(), d_grad_b, gpu_b.size() * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (std::size_t i = 0; i < gpu_w.size(); ++i)
    if (std::fabs(cpu_w[i] - gpu_w[i]) > 1.0e-6f)
      ok = false;
  for (std::size_t i = 0; i < gpu_b.size(); ++i)
    if (std::fabs(cpu_b[i] - gpu_b[i]) > 1.0e-6f)
      ok = false;
  std::cout << "Bias gradients:";
  for (float value : gpu_b)
    std::cout << " " << value;
  std::cout << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_hidden));
  CHECK_CUDA(cudaFree(d_prediction));
  CHECK_CUDA(cudaFree(d_target));
  CHECK_CUDA(cudaFree(d_grad_w));
  CHECK_CUDA(cudaFree(d_grad_b));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
