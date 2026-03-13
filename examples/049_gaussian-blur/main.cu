// Example 049: Gaussian Blur
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

__global__ void gaussian_kernel(const float *input, float *output, int w, int h) {
  __shared__ float k[9];
  if (threadIdx.x < 9 && threadIdx.y == 0) {
    float vals[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    k[threadIdx.x] = vals[threadIdx.x] / 16.0f;
  }
  __syncthreads();
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    float sum = 0.0f;
    for (int ky = -1; ky <= 1; ++ky)
      for (int kx = -1; kx <= 1; ++kx) {
        int sx = min(max(x + kx, 0), w - 1);
        int sy = min(max(y + ky, 0), h - 1);
        sum += input[sy * w + sx] * k[(ky + 1) * 3 + (kx + 1)];
      }
    output[y * w + x] = sum;
  }
}
int main() {
  const int w = 8, h = 8;
  std::vector<float> in(w * h), gpu(w * h, 0.0f), cpu(w * h, 0.0f),
      k = {1 / 16.0f, 2 / 16.0f, 1 / 16.0f, 2 / 16.0f, 4 / 16.0f,
           2 / 16.0f, 1 / 16.0f, 2 / 16.0f, 1 / 16.0f};
  for (int i = 0; i < w * h; ++i)
    in[i] = (float)((i % 5) + 1);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx) {
          int sx = std::min(std::max(x + kx, 0), w - 1);
          int sy = std::min(std::max(y + ky, 0), h - 1);
          cpu[y * w + x] += in[sy * w + sx] * k[(ky + 1) * 3 + (kx + 1)];
        }
  float *di = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, in.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&do_, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(di, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 t(16, 16), bl((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
  gaussian_kernel<<<bl, t>>>(di, do_, w, h);
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
