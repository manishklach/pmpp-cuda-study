// Example 050: Median Filter
// Difficulty: Intermediate

// Track: Linear Algebra
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#define CHECK_CUDA(call)                                                                       \
  do {                                                                                         \
    cudaError_t status__ = (call);                                                             \
    if (status__ != cudaSuccess) {                                                             \
      std::cerr << "CUDA error: " << cudaGetErrorString(status__) << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;                                                      \
      std::exit(EXIT_FAILURE);                                                                 \
    }                                                                                          \
  } while (0)

__device__ void sort9(float *vals) {
  for (int i = 0; i < 9; ++i)
    for (int j = i + 1; j < 9; ++j)
      if (vals[j] < vals[i]) {
        float t = vals[i];
        vals[i] = vals[j];
        vals[j] = t;
      }
}
__global__ void median3x3_kernel(const float *input, float *output, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    float vals[9];
    int p = 0;
    for (int ky = -1; ky <= 1; ++ky)
      for (int kx = -1; kx <= 1; ++kx) {
        int sx = min(max(x + kx, 0), w - 1);
        int sy = min(max(y + ky, 0), h - 1);
        vals[p++] = input[sy * w + sx];
      }
    sort9(vals);
    output[y * w + x] = vals[4];
  }
}
int main() {
  const int w = 8, h = 8;
  std::vector<float> in(w * h), gpu(w * h, 0.0f), cpu(w * h, 0.0f);
  for (int i = 0; i < w * h; ++i)
    in[i] = (float)((i * 3) % 11);
  in[10] = 99.0f;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float vals[9];
      int p = 0;
      for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx) {
          int sx = std::min(std::max(x + kx, 0), w - 1);
          int sy = std::min(std::max(y + ky, 0), h - 1);
          vals[p++] = in[sy * w + sx];
        }
      std::sort(vals, vals + 9);
      cpu[y * w + x] = vals[4];
    }
  float *di = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, in.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&do_, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(di, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 t(16, 16), bl((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
  median3x3_kernel<<<bl, t>>>(di, do_, w, h);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), do_, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(gpu[i] - cpu[i]) > 1e-5f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(do_));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
