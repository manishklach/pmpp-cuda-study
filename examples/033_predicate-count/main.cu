// Example 033: Predicate Count

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

__global__ void count_positive_kernel(const int *input, int *count, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && input[idx] > 0)
    atomicAdd(count, 1);
}
int main() {
  const int n = 1024;
  std::vector<int> input(n);
  int cpu = 0;
  for (int i = 0; i < n; ++i) {
    input[i] = (i % 11) - 5;
    if (input[i] > 0)
      ++cpu;
  }
  int *di = nullptr, *dc = nullptr;
  int gpu = 0;
  CHECK_CUDA(cudaMalloc(&di, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dc, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(di, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dc, 0, sizeof(int)));
  count_positive_kernel<<<(n + 255) / 256, 256>>>(di, dc, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(&gpu, dc, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "Validation: " << (gpu == cpu ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(dc));
  return gpu == cpu ? EXIT_SUCCESS : EXIT_FAILURE;
}
