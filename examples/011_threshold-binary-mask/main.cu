// Example 011: Threshold Binary Mask
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
__global__ void k(const float *in, unsigned char *mask, float th, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    mask[idx] = in[idx] >= th ? 1u : 0u;
}
int main() {
  const int n = 64;
  const float th = 0.0f;
  size_t ib = (size_t)n * sizeof(float), mb = (size_t)n * sizeof(unsigned char);
  std::vector<float> in(n);
  std::vector<unsigned char> go(n, 0), co(n, 0);
  for (int i = 0; i < n; ++i) {
    in[i] = (i % 9) - 4;
    co[i] = in[i] >= th ? 1u : 0u;
  }
  float *di = nullptr;
  unsigned char *dm = nullptr;
  CHECK_CUDA(cudaMalloc(&di, ib));
  CHECK_CUDA(cudaMalloc(&dm, mb));
  CHECK_CUDA(cudaMemcpy(di, in.data(), ib, cudaMemcpyHostToDevice));
  k<<<1, 128>>>(di, dm, th, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(go.data(), dm, mb, cudaMemcpyDeviceToHost));
  bool ok = go == co;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(dm));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
