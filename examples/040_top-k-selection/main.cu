// Example 040: Top K Selection

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

__global__ void bitonic_step_kernel(int *data, int j, int k) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int ixj = i ^ j;
  if (ixj > i) {
    bool ascending = (i & k) == 0;
    if ((ascending && data[i] > data[ixj]) || (!ascending && data[i] < data[ixj])) {
      int t = data[i];
      data[i] = data[ixj];
      data[ixj] = t;
    }
  }
}
int main() {
  const int n = 32, k_top = 5;
  std::vector<int> input = {12, 99, 3, 47, 18, 76, 5,  65, 23, 88, 14, 54, 67, 31, 42, 90,
                            1,  72, 8, 60, 27, 81, 36, 95, 11, 58, 69, 20, 84, 7,  52, 40};
  auto cpu = input;
  std::sort(cpu.begin(), cpu.end(), std::greater<int>());
  std::vector<int> cpu_top(cpu.begin(), cpu.begin() + k_top);
  int *d = nullptr;
  CHECK_CUDA(cudaMalloc(&d, n * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  for (int k = 2; k <= n; k <<= 1)
    for (int j = k >> 1; j > 0; j >>= 1) {
      bitonic_step_kernel<<<1, 128>>>(d, j, k);
      CHECK_CUDA(cudaGetLastError());
    }
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<int> sorted(n);
  CHECK_CUDA(cudaMemcpy(sorted.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
  std::reverse(sorted.begin(), sorted.end());
  std::vector<int> gpu_top(sorted.begin(), sorted.begin() + k_top);
  bool ok = gpu_top == cpu_top;
  std::cout << "Top-k: ";
  for (int v : gpu_top)
    std::cout << v << ' ';
  std::cout << "\nValidation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
