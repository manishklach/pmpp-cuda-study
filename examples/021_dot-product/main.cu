// Example 021: Dot Product

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

__global__ void dot_partials_kernel(const float *a, const float *b, float *partials, int n) {
  __shared__ float scratch[256];
  int global = blockIdx.x * blockDim.x + threadIdx.x;
  int local = threadIdx.x;
  float value = global < n ? a[global] * b[global] : 0.0f;
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
  const int n = 1 << 12;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  std::vector<float> a(n), b(n), partials(blocks, 0.0f);
  float cpu = 0.0f;
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>((i % 17) - 8) * 0.25f;
    b[i] = static_cast<float>((i % 11) - 5) * 0.5f;
    cpu += a[i] * b[i];
  }
  float *da = nullptr, *db = nullptr, *dp = nullptr;
  CHECK_CUDA(cudaMalloc(&da, bytes));
  CHECK_CUDA(cudaMalloc(&db, bytes));
  CHECK_CUDA(cudaMalloc(&dp, blocks * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
  dot_partials_kernel<<<blocks, threads>>>(da, db, dp, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(partials.data(), dp, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  float gpu = std::accumulate(partials.begin(), partials.end(), 0.0f);
  std::cout << "CPU dot: " << cpu << "\nGPU dot: " << gpu << std::endl;
  std::cout << "Validation: " << (std::fabs(cpu - gpu) < 1.0e-3f ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(dp));
  return std::fabs(cpu - gpu) < 1.0e-3f ? EXIT_SUCCESS : EXIT_FAILURE;
}
