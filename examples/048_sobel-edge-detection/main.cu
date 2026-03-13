// Example 048: Sobel Edge Detection
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

__global__ void sobel_kernel(const float *input, float *output, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x > 0 && x + 1 < w && y > 0 && y + 1 < h) {
    int idx = y * w + x;
    float gx = -input[(y - 1) * w + (x - 1)] + input[(y - 1) * w + (x + 1)] -
               2 * input[y * w + (x - 1)] + 2 * input[y * w + (x + 1)] -
               input[(y + 1) * w + (x - 1)] + input[(y + 1) * w + (x + 1)];
    float gy = -input[(y - 1) * w + (x - 1)] - 2 * input[(y - 1) * w + x] -
               input[(y - 1) * w + (x + 1)] + input[(y + 1) * w + (x - 1)] +
               2 * input[(y + 1) * w + x] + input[(y + 1) * w + (x + 1)];
    output[idx] = sqrtf(gx * gx + gy * gy);
  }
}
int main() {
  const int w = 8, h = 8;
  std::vector<float> in(w * h), gpu(w * h, 0.0f), cpu(w * h, 0.0f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      in[y * w + x] = (x < 4 ? 0.0f : 10.0f);
  for (int y = 1; y < h - 1; ++y)
    for (int x = 1; x < w - 1; ++x) {
      float gx = -in[(y - 1) * w + (x - 1)] + in[(y - 1) * w + (x + 1)] - 2 * in[y * w + (x - 1)] +
                 2 * in[y * w + (x + 1)] - in[(y + 1) * w + (x - 1)] + in[(y + 1) * w + (x + 1)];
      float gy = -in[(y - 1) * w + (x - 1)] - 2 * in[(y - 1) * w + x] - in[(y - 1) * w + (x + 1)] +
                 in[(y + 1) * w + (x - 1)] + 2 * in[(y + 1) * w + x] + in[(y + 1) * w + (x + 1)];
      cpu[y * w + x] = std::sqrt(gx * gx + gy * gy);
    }
  float *di = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&di, in.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&do_, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(di, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 t(16, 16), bl((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
  sobel_kernel<<<bl, t>>>(di, do_, w, h);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), do_, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(gpu[i] - cpu[i]) > 1e-4f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di));
  CHECK_CUDA(cudaFree(do_));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
