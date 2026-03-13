// Example 045: Convolution 1D
// Difficulty: Intermediate

// Track: Linear Algebra
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#define CHECK_CUDA(call)                                                                       \
  do {                                                                                         \
    cudaError_t status__ = (call);                                                             \
    if (status__ != cudaSuccess) {                                                             \
      std::cerr << "CUDA error: " << cudaGetErrorString(status__) << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;                                                      \
      std::exit(EXIT_FAILURE);                                                                 \
    }                                                                                          \
  } while (0)

__global__ void conv1d_kernel(const float *input, const float *kernel, float *output, int n,
                              int radius) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
      int x = min(max(idx + k, 0), n - 1);
      sum += input[x] * kernel[k + radius];
    }
    output[idx] = sum;
  }
}
int main() {
  const int n = 64, r = 1;
  std::vector<float> in(n), ker = {0.25f, 0.5f, 0.25f}, gpu(n, 0.0f), cpu(n, 0.0f);
  for (int i = 0; i < n; ++i)
    in[i] = (float)(i % 9);
  for (int i = 0; i < n; ++i)
    for (int k = -r; k <= r; ++k) {
      int x = std::min(std::max(i + k, 0), n - 1);
      cpu[i] += in[x] * ker[k + r];
    }
  float *di = nullptr, *dk = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dk, ker.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&do_, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(di, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dk, ker.data(), ker.size() * sizeof(float), cudaMemcpyHostToDevice));
  conv1d_kernel<<<1, 128>>>(di, dk, do_, n, r);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), do_, n * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (std::fabs(gpu[i] - cpu[i]) > 1e-5f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(dk));
  CHECK_CUDA(cudaFree(do_));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
