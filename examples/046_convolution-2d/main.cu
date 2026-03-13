// Example 046: Convolution 2D
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

__global__ void conv2d_kernel(const float *input, const float *kernel, float *output, int w, int h,
                              int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky) {
      for (int kx = -radius; kx <= radius; ++kx) {
        int sx = min(max(x + kx, 0), w - 1);
        int sy = min(max(y + ky, 0), h - 1);
        sum += input[sy * w + sx] * kernel[(ky + radius) * (2 * radius + 1) + (kx + radius)];
      }
    }
    output[y * w + x] = sum;
  }
}
int main() {
  const int w = 8, h = 8, r = 1;
  std::vector<float> in(w * h), ker = {0, 1, 0, 1, 4, 1, 0, 1, 0}, gpu(w * h, 0.0f),
                                cpu(w * h, 0.0f);
  for (int i = 0; i < w * h; ++i)
    in[i] = (float)((i % 7) + 1);
  for (float &k : ker)
    k /= 8.0f;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int ky = -r; ky <= r; ++ky)
        for (int kx = -r; kx <= r; ++kx) {
          int sx = std::min(std::max(x + kx, 0), w - 1);
          int sy = std::min(std::max(y + ky, 0), h - 1);
          cpu[y * w + x] += in[sy * w + sx] * ker[(ky + r) * 3 + (kx + r)];
        }
  float *di = nullptr, *dk = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, in.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dk, ker.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&do_, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(di, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dk, ker.data(), ker.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 t(16, 16), bl((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
  conv2d_kernel<<<bl, t>>>(di, dk, do_, w, h, r);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), do_, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(gpu[i] - cpu[i]) > 1e-5f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(dk));
  CHECK_CUDA(cudaFree(do_));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
