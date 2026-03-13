// Example 032: Scatter

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

__global__ void scatter_kernel(const float *input, const int *destinations, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    output[destinations[idx]] = input[idx];
}
int main() {
  const int n = 32;
  std::vector<float> input(n), gpu(n, -1.0f), cpu(n, -1.0f);
  std::vector<int> dst(n);
  for (int i = 0; i < n; ++i) {
    input[i] = 100.0f + i;
    dst[i] = (i * 5) % n;
    cpu[dst[i]] = input[i];
  }
  float *di = nullptr, *do_ = nullptr;
  int *dd = nullptr;
  CHECK_CUDA(cudaMalloc(&di, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&do_, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dd, n * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(di, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dd, dst.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(do_, 0, n * sizeof(float)));
  scatter_kernel<<<1, 128>>>(di, dd, do_, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), do_, n * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (std::fabs(gpu[i] - cpu[i]) > 1e-5f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(do_));
  CHECK_CUDA(cudaFree(dd));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
