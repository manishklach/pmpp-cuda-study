// Example 031: Gather

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

__global__ void gather_kernel(const float *source, const int *indices, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    output[idx] = source[indices[idx]];
}
int main() {
  const int n = 32;
  std::vector<float> source(64), gpu(n, 0.0f), cpu(n, 0.0f);
  std::vector<int> idxs(n);
  for (int i = 0; i < 64; ++i)
    source[i] = i * 1.5f;
  for (int i = 0; i < n; ++i) {
    idxs[i] = (i * 3) % 64;
    cpu[i] = source[idxs[i]];
  }
  float *ds = nullptr, *do_ = nullptr;
  int *di = nullptr;
  CHECK_CUDA(cudaMalloc(&ds, 64 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&di, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&do_, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(ds, source.data(), 64 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(di, idxs.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  gather_kernel<<<1, 128>>>(ds, di, do_, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), do_, n * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (std::fabs(gpu[i] - cpu[i]) > 1e-5f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(ds));
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(do_));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
