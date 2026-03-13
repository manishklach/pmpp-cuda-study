// Example 061: Image Resize Nearest Neighbor
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

__global__ void resize_nearest_kernel(const float *src, int src_width, int src_height, float *dst,
                                      int dst_width, int dst_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_width || y >= dst_height)
    return;

  float scale_x = static_cast<float>(src_width) / static_cast<float>(dst_width);
  float scale_y = static_cast<float>(src_height) / static_cast<float>(dst_height);
  int src_x = min(static_cast<int>(x * scale_x), src_width - 1);
  int src_y = min(static_cast<int>(y * scale_y), src_height - 1);
  dst[y * dst_width + x] = src[src_y * src_width + src_x];
}

int main() {
  const int src_width = 6;
  const int src_height = 4;
  const int dst_width = 12;
  const int dst_height = 8;
  std::vector<float> src(src_width * src_height), cpu(dst_width * dst_height, 0.0f),
      gpu(dst_width * dst_height, 0.0f);
  for (int y = 0; y < src_height; ++y)
    for (int x = 0; x < src_width; ++x)
      src[y * src_width + x] = static_cast<float>(y * 10 + x);
  for (int y = 0; y < dst_height; ++y)
    for (int x = 0; x < dst_width; ++x) {
      int src_x = std::min(static_cast<int>(x * (static_cast<float>(src_width) / dst_width)),
                           src_width - 1);
      int src_y = std::min(static_cast<int>(y * (static_cast<float>(src_height) / dst_height)),
                           src_height - 1);
      cpu[y * dst_width + x] = src[src_y * src_width + src_x];
    }

  float *d_src = nullptr, *d_dst = nullptr;
  CHECK_CUDA(cudaMalloc(&d_src, src.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_dst, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_src, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 threads(16, 16);
  dim3 blocks((dst_width + threads.x - 1) / threads.x, (dst_height + threads.y - 1) / threads.y);
  resize_nearest_kernel<<<blocks, threads>>>(d_src, src_width, src_height, d_dst, dst_width,
                                             dst_height);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_dst, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (std::size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-6f)
      ok = false;
  std::cout << "Output size: " << dst_width << "x" << dst_height << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_src));
  CHECK_CUDA(cudaFree(d_dst));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
