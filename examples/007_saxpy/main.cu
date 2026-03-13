// Example 007: SAXPY
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
__global__ void k(float a, const float *x, const float *y, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = a * x[idx] + y[idx];
}
int main() {
  const int n = 1 << 13;
  const float a = 1.75f;
  size_t bytes = (size_t)n * sizeof(float);
  std::vector<float> x(n), y(n), go(n, 0.0f), co(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    x[i] = (i % 31) * 0.125f;
    y[i] = ((i % 11) - 5) * 0.5f;
    co[i] = a * x[i] + y[i];
  }
  float *dx = nullptr, *dy = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&dx, bytes));
  CHECK_CUDA(cudaMalloc(&dy, bytes));
  CHECK_CUDA(cudaMalloc(&do_, bytes));
  CHECK_CUDA(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dy, y.data(), bytes, cudaMemcpyHostToDevice));
  k<<<(n + 255) / 256, 256>>>(a, dx, dy, do_, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(go.data(), do_, bytes, cudaMemcpyDeviceToHost));
  int mm = 0;
  for (int i = 0; i < n; ++i)
    if (fabs(go[i] - co[i]) > 1.0e-5f)
      ++mm;
  std::cout << "Validation: " << (mm == 0 ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(dx));
  CHECK_CUDA(cudaFree(dy));
  CHECK_CUDA(cudaFree(do_));
  return mm == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
