// Example 026: Prefix Sum Naive Scan

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

__global__ void hillis_steele_kernel(const int *input, int *output, int n) {
  __shared__ int data[256];
  int tid = threadIdx.x;
  data[tid] = tid < n ? input[tid] : 0;
  __syncthreads();
  for (int offset = 1; offset < n; offset <<= 1) {
    int add = tid >= offset ? data[tid - offset] : 0;
    __syncthreads();
    if (tid < n)
      data[tid] += add;
    __syncthreads();
  }
  if (tid < n)
    output[tid] = data[tid];
}
int main() {
  const int n = 128;
  std::vector<int> input(n), gpu(n, 0), cpu(n, 0);
  for (int i = 0; i < n; ++i) {
    input[i] = (i % 5) + 1;
    cpu[i] = input[i] + (i ? cpu[i - 1] : 0);
  }
  int *di = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&do_, n * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(di, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  hillis_steele_kernel<<<1, 256>>>(di, do_, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), do_, n * sizeof(int), cudaMemcpyDeviceToHost));
  bool ok = gpu == cpu;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(do_));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
