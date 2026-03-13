// Example 022: L2 Norm

// Track: Parallel Patterns
// Difficulty: Intermediate
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#define CHECK_CUDA(call)                                                                           \
  do {                                                                                             \
    cudaError_t status__ = (call);                                                                 \
    if (status__ != cudaSuccess) {                                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(status__) << " at " << __FILE__ << ":"     \
                << __LINE__ << std::endl;                                                          \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

__global__ void squared_sum_partials_kernel(const float *x, float *partials, int n) {
  __shared__ float scratch[256];
  int global = blockIdx.x * blockDim.x + threadIdx.x;
  int local = threadIdx.x;
  float value = global < n ? x[global] * x[global] : 0.0f;
  scratch[local] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local < stride)
      scratch[local] += scratch[local + stride];
    __syncthreads();
  }
  if (local == 0)
    partials[blockIdx.x] = scratch[0];
}

int main() {
  const int n = 2048, threads = 256, blocks = (n + threads - 1) / threads;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  std::vector<float> x(n), partials(blocks, 0.0f);
  float cpu_sq = 0.0f;
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>((i % 21) - 10) * 0.125f;
    cpu_sq += x[i] * x[i];
  }
  float *dx = nullptr, *dp = nullptr;
  CHECK_CUDA(cudaMalloc(&dx, bytes));
  CHECK_CUDA(cudaMalloc(&dp, blocks * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice));
  squared_sum_partials_kernel<<<blocks, threads>>>(dx, dp, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(partials.data(), dp, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  float gpu_sq = std::accumulate(partials.begin(), partials.end(), 0.0f);
  float cpu = std::sqrt(cpu_sq), gpu = std::sqrt(gpu_sq);
  std::cout << "CPU L2: " << cpu << "\nGPU L2: " << gpu << std::endl;
  std::cout << "Validation: " << (std::fabs(cpu - gpu) < 1.0e-3f ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(dx));
  CHECK_CUDA(cudaFree(dp));
  return std::fabs(cpu - gpu) < 1.0e-3f ? EXIT_SUCCESS : EXIT_FAILURE;
}
