// Example 016: 1D Stencil
// Track: Foundations
// Difficulty: Intermediate
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
__global__ void k(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float l = in[idx > 0 ? idx - 1 : idx], c = in[idx], r = in[idx + 1 < n ? idx + 1 : idx];
    out[idx] = (l + c + r) / 3.0f;
  }
}
int main() {
  const int n = 64;
  size_t bytes = (size_t)n * sizeof(float);
  std::vector<float> in(n), go(n, 0.0f), co(n, 0.0f);
  for (int i = 0; i < n; ++i)
    in[i] = (float)i;
  for (int i = 0; i < n; ++i) {
    float l = in[i > 0 ? i - 1 : i], c = in[i], r = in[i + 1 < n ? i + 1 : i];
    co[i] = (l + c + r) / 3.0f;
  }
  float *di = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, bytes));
  CHECK_CUDA(cudaMalloc(&do_, bytes));
  CHECK_CUDA(cudaMemcpy(di, in.data(), bytes, cudaMemcpyHostToDevice));
  k<<<1, 128>>>(di, do_, n);
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
