// Example 030: Stream Compaction

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

__global__ void compact_positive_kernel(const int *input, int *output, int *count, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && input[idx] > 0) {
    int slot = atomicAdd(count, 1);
    output[slot] = input[idx];
  }
}
int main() {
  const int n = 64;
  std::vector<int> input(n), cpu;
  for (int i = 0; i < n; ++i) {
    input[i] = (i % 9) - 4;
    if (input[i] > 0)
      cpu.push_back(input[i]);
  }
  std::vector<int> gpu(cpu.size(), 0);
  int *di = nullptr, *do_ = nullptr, *dc = nullptr;
  CHECK_CUDA(cudaMalloc(&di, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&do_, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dc, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(di, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dc, 0, sizeof(int)));
  compact_positive_kernel<<<1, 128>>>(di, do_, dc, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  int count = 0;
  CHECK_CUDA(cudaMemcpy(&count, dc, sizeof(int), cudaMemcpyDeviceToHost));
  gpu.resize(count);
  CHECK_CUDA(cudaMemcpy(gpu.data(), do_, count * sizeof(int), cudaMemcpyDeviceToHost));
  bool ok = gpu == cpu;
  std::cout << "Kept " << count << " elements\nValidation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(do_));
  CHECK_CUDA(cudaFree(dc));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
