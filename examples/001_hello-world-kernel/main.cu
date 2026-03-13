// Example 001: Hello World Kernel
// Track: Foundations
// Difficulty: Beginner
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
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
#include <cstdio>
__global__ void hello_kernel(int *ids) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  ids[tid] = tid;
  printf("Hello from block %d thread %d (linear %d)\n", blockIdx.x, threadIdx.x, tid);
}
int main() {
  const int blocks = 2, threads = 4, total = blocks * threads;
  int *d = nullptr;
  std::vector<int> ids(total, -1);
  CHECK_CUDA(cudaMalloc(&d, total * sizeof(int)));
  hello_kernel<<<blocks, threads>>>(d);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(ids.data(), d, total * sizeof(int), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < total; ++i)
    if (ids[i] != i)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
