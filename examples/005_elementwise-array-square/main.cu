// Example 005: Elementwise Array Square
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
__global__ void kernel(const float *x, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = x[idx] * x[idx];
}
int main() {
  const int n = 1 << 12;
  size_t bytes = (size_t)n * sizeof(float);
  std::vector<float> in(n), go(n, 0.0f), co(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    float value = ((i % 23) - 11) * 0.25f;
    in[i] = value;
    co[i] = value * value;
  }
  float *di = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, bytes));
  CHECK_CUDA(cudaMalloc(&do_, bytes));
  CHECK_CUDA(cudaMemcpy(di, in.data(), bytes, cudaMemcpyHostToDevice));
  kernel<<<(n + 255) / 256, 256>>>(di, do_, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(go.data(), do_, bytes, cudaMemcpyDeviceToHost));
  int mm = 0;
  for (int i = 0; i < n; ++i)
    if (fabs(go[i] - co[i]) > 1.0e-5f)
      ++mm;
  std::cout << "Validation: " << (mm == 0 ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(do_));
  return mm == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
