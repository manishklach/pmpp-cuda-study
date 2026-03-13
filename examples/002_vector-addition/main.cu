// Example 002: Vector Addition
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
__global__ void kernel(const float *a, const float *b, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = a[idx] + b[idx];
}
int main() {
  const int n = 1 << 12;
  size_t bytes = (size_t)n * sizeof(float);
  std::vector<float> a(n), b(n), go(n, 0.0f), co(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    float lhs = ((i % 29) - 14) * 0.25f;
    float rhs = ((i % 13) - 6) * 0.5f;
    a[i] = lhs;
    b[i] = rhs;
    co[i] = lhs + rhs;
  }
  float *da = nullptr, *db = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&da, bytes));
  CHECK_CUDA(cudaMalloc(&db, bytes));
  CHECK_CUDA(cudaMalloc(&do_, bytes));
  CHECK_CUDA(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
  kernel<<<(n + 255) / 256, 256>>>(da, db, do_, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(go.data(), do_, bytes, cudaMemcpyDeviceToHost));
  int mm = 0;
  for (int i = 0; i < n; ++i)
    if (fabs(go[i] - co[i]) > 1.0e-5f)
      ++mm;
  std::cout << "Validation: " << (mm == 0 ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(do_));
  return mm == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
