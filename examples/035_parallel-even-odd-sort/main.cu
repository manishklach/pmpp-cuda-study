// Example 035: Parallel Even Odd Sort

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

__global__ void odd_even_phase_kernel(int *data, int n, int phase) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int i = 2 * tid + phase;
  if (i + 1 < n && data[i] > data[i + 1]) {
    int t = data[i];
    data[i] = data[i + 1];
    data[i + 1] = t;
  }
}
int main() {
  const int n = 32;
  std::vector<int> input = {9,  4,  1,  7,  3,  8,  2,  6,  5,  0,  11, 10, 13, 12, 15, 14,
                            19, 16, 18, 17, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30};
  auto cpu = input;
  std::sort(cpu.begin(), cpu.end());
  int *d = nullptr;
  CHECK_CUDA(cudaMalloc(&d, n * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  for (int phase = 0; phase < n; ++phase) {
    odd_even_phase_kernel<<<1, 128>>>(d, n, phase & 1);
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
