// Example 034: Find First Match

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

__global__ void find_first_kernel(const int *input, int target, int *first_index, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && input[idx] == target)
    atomicMin(first_index, idx);
}
int main() {
  const int n = 512;
  const int target = 42;
  std::vector<int> input(n);
  std::fill(input.begin(), input.end(), 7);
  input[137] = target;
  input[299] = target;
  int cpu = 137, gpu = n;
  int *di = nullptr, *df = nullptr;
  CHECK_CUDA(cudaMalloc(&di, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&df, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(di, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(df, &gpu, sizeof(int), cudaMemcpyHostToDevice));
  find_first_kernel<<<(n + 255) / 256, 256>>>(di, target, df, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(&gpu, df, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "Validation: " << (gpu == cpu ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(df));
  return gpu == cpu ? EXIT_SUCCESS : EXIT_FAILURE;
}
