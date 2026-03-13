// Example 047: Separable Convolution
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

__global__ void conv_horizontal_kernel(const float *input, const float *kernel, float *output,
                                       int w, int h, int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
      int sx = min(max(x + k, 0), w - 1);
      sum += input[y * w + sx] * kernel[k + radius];
    }
    output[y * w + x] = sum;
  }
}
__global__ void conv_vertical_kernel(const float *input, const float *kernel, float *output, int w,
                                     int h, int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
      int sy = min(max(y + k, 0), h - 1);
      sum += input[sy * w + x] * kernel[k + radius];
    }
    output[y * w + x] = sum;
  }
}
int main() {
  const int w = 8, h = 8, r = 1;
  std::vector<float> in(w * h), ker = {0.25f, 0.5f, 0.25f}, temp(w * h, 0.0f), gpu(w * h, 0.0f),
                                cpu(w * h, 0.0f), cpu_temp(w * h, 0.0f);
  for (int i = 0; i < w * h; ++i)
    in[i] = (float)((i % 9) + 1);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int k = -r; k <= r; ++k) {
        int sx = std::min(std::max(x + k, 0), w - 1);
        cpu_temp[y * w + x] += in[y * w + sx] * ker[k + r];
      }
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int k = -r; k <= r; ++k) {
        int sy = std::min(std::max(y + k, 0), h - 1);
        cpu[y * w + x] += cpu_temp[sy * w + x] * ker[k + r];
      }
  float *di = nullptr, *dk = nullptr, *dt = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, in.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dk, ker.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dt, temp.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&do_, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(di, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dk, ker.data(), ker.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 t(16, 16), bl((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
  conv_horizontal_kernel<<<bl, t>>>(di, dk, dt, w, h, r);
  conv_vertical_kernel<<<bl, t>>>(dt, dk, do_, w, h, r);
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
  CHECK_CUDA(cudaFree(dt));
  CHECK_CUDA(cudaFree(do_));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
