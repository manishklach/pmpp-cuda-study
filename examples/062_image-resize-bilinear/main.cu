// Example 062: Image Resize Bilinear
// Track: Image and Signal
// Difficulty: Intermediate
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

inline void check_cuda(cudaError_t status, const char *file, int line) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << file << ":" << line
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

__device__ float lerp(float a, float b, float t) {
  return a + (b - a) * t;
}

__global__ void resize_bilinear_kernel(const float *src, int src_width, int src_height, float *dst,
                                       int dst_width, int dst_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_width || y >= dst_height)
    return;
  float gx = (static_cast<float>(x) + 0.5f) * src_width / dst_width - 0.5f;
  float gy = (static_cast<float>(y) + 0.5f) * src_height / dst_height - 0.5f;
  int x0 = max(0, min(static_cast<int>(floorf(gx)), src_width - 1));
  int y0 = max(0, min(static_cast<int>(floorf(gy)), src_height - 1));
  int x1 = min(x0 + 1, src_width - 1);
  int y1 = min(y0 + 1, src_height - 1);
  float tx = gx - x0;
  float ty = gy - y0;
  float top = lerp(src[y0 * src_width + x0], src[y0 * src_width + x1], tx);
  float bottom = lerp(src[y1 * src_width + x0], src[y1 * src_width + x1], tx);
  dst[y * dst_width + x] = lerp(top, bottom, ty);
}

int main() {
  const int src_width = 5, src_height = 5, dst_width = 9, dst_height = 7;
  std::vector<float> src(src_width * src_height), cpu(dst_width * dst_height, 0.0f),
      gpu(dst_width * dst_height, 0.0f);
  for (int y = 0; y < src_height; ++y)
    for (int x = 0; x < src_width; ++x)
      src[y * src_width + x] = static_cast<float>((x + 1) * (y + 2));
  for (int y = 0; y < dst_height; ++y)
    for (int x = 0; x < dst_width; ++x) {
      float gx = (static_cast<float>(x) + 0.5f) * src_width / dst_width - 0.5f;
      float gy = (static_cast<float>(y) + 0.5f) * src_height / dst_height - 0.5f;
      int x0 = std::max(0, std::min(static_cast<int>(std::floor(gx)), src_width - 1));
      int y0 = std::max(0, std::min(static_cast<int>(std::floor(gy)), src_height - 1));
      int x1 = std::min(x0 + 1, src_width - 1);
      int y1 = std::min(y0 + 1, src_height - 1);
      float tx = gx - x0;
      float ty = gy - y0;
      float top =
          src[y0 * src_width + x0] + (src[y0 * src_width + x1] - src[y0 * src_width + x0]) * tx;
      float bottom =
          src[y1 * src_width + x0] + (src[y1 * src_width + x1] - src[y1 * src_width + x0]) * tx;
      cpu[y * dst_width + x] = top + (bottom - top) * ty;
    }

  float *d_src = nullptr, *d_dst = nullptr;
  CHECK_CUDA(cudaMalloc(&d_src, src.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_dst, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_src, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 threads(16, 16);
  dim3 blocks((dst_width + threads.x - 1) / threads.x, (dst_height + threads.y - 1) / threads.y);
  resize_bilinear_kernel<<<blocks, threads>>>(d_src, src_width, src_height, d_dst, dst_width,
                                              dst_height);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_dst, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (std::size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-4f)
      ok = false;
  std::cout << "Interpolation samples: " << gpu[0] << ", " << gpu[dst_width + 1] << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_src));
  CHECK_CUDA(cudaFree(d_dst));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
