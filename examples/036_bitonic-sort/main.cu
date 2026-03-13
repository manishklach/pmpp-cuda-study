// Example 036: Bitonic Sort

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
  const int n = 32;
  std::vector<int> input = {23, 1, 17, 9, 3, 15, 7,  13, 31, 29, 27, 25, 21, 19, 11, 5,
                            0,  2, 4,  6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
  auto cpu = input;
  std::sort(cpu.begin(), cpu.end());
  int *d = nullptr;
  CHECK_CUDA(cudaMalloc(&d, n * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  for (int k = 2; k <= n; k <<= 1)
    for (int j = k >> 1; j > 0; j >>= 1) {
      bitonic_step_kernel<<<1, 128>>>(d, j, k);
      CHECK_CUDA(cudaGetLastError());
    }
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<int> gpu(n);
  CHECK_CUDA(cudaMemcpy(gpu.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
  bool ok = gpu == cpu;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
