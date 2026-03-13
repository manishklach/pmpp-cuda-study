// Example 010: Clamp Values To Range
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
__global__ void k(const float *in, float *out, float lo, float hi, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float v = in[idx];
    v = v < lo ? lo : v;
    v = v > hi ? hi : v;
    out[idx] = v;
  }
}
int main() {
  const int n = 128;
  const float lo = -1.5f, hi = 2.0f;
  size_t bytes = (size_t)n * sizeof(float);
  std::vector<float> in(n), go(n, 0.0f), co(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    in[i] = 0.25f * ((i % 31) - 15);
    co[i] = in[i] < lo ? lo : (in[i] > hi ? hi : in[i]);
  }
  float *di = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, bytes));
  CHECK_CUDA(cudaMalloc(&do_, bytes));
  CHECK_CUDA(cudaMemcpy(di, in.data(), bytes, cudaMemcpyHostToDevice));
  k<<<1, 256>>>(di, do_, lo, hi, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(go.data(), do_, bytes, cudaMemcpyDeviceToHost));
  int mm = 0;
  for (int i = 0; i < n; ++i)
    if (fabs(go[i] - co[i]) > 1.0e-6f)
      ++mm;
  std::cout << "Validation: " << (mm == 0 ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(do_));
  return mm == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
