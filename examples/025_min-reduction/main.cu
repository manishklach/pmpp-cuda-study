// Example 025: Min Reduction

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

__global__ void min_partials_kernel(const float *input, float *partials, int n) {
  __shared__ float scratch[256];
  int global = blockIdx.x * blockDim.x + threadIdx.x;
  int local = threadIdx.x;
  scratch[local] = global < n ? input[global] : FLT_MAX;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local < stride)
      scratch[local] = fminf(scratch[local], scratch[local + stride]);
    __syncthreads();
  }
  if (local == 0)
    partials[blockIdx.x] = scratch[0];
}

int main() {
  const int n = 2048, threads = 256, blocks = (n + threads - 1) / threads;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  std::vector<float> input(n), partials(blocks, 0.0f);
  for (int i = 0; i < n; ++i)
    input[i] = static_cast<float>((i % 41) - 20);
  input[333] = -999.0f;
  float cpu = *std::min_element(input.begin(), input.end());
  float *di = nullptr, *dp = nullptr;
  CHECK_CUDA(cudaMalloc(&di, bytes));
  CHECK_CUDA(cudaMalloc(&dp, blocks * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(di, input.data(), bytes, cudaMemcpyHostToDevice));
  min_partials_kernel<<<blocks, threads>>>(di, dp, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(partials.data(), dp, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  float gpu = *std::min_element(partials.begin(), partials.end());
  std::cout << "Validation: " << (std::fabs(cpu - gpu) < 1.0e-5f ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(dp));
  return std::fabs(cpu - gpu) < 1.0e-5f ? EXIT_SUCCESS : EXIT_FAILURE;
}
