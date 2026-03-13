// Example 082: Heat Diffusion Grid
// Track: Simulation
// Difficulty: Intermediate
// Status: Reference-friendly

#include <cuda_runtime.h>
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

__global__ void heat_step_kernel(const float *current, float *next, int width, int height,
                                 float alpha) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  int idx = y * width + x;
  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    next[idx] = current[idx];
    return;
  }
  float center = current[idx];
  float laplacian = current[idx - 1] + current[idx + 1] + current[idx - width] +
                    current[idx + width] - 4.0f * center;
  next[idx] = center + alpha * laplacian;
}

int main() {
  const int width = 8, height = 8;
  const float alpha = 0.2f;
  std::vector<float> current(width * height, 0.0f), cpu(width * height, 0.0f),
      gpu(width * height, 0.0f);
  current[(height / 2) * width + width / 2] = 100.0f;
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      int idx = y * width + x;
      if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
        cpu[idx] = current[idx];
      else {
        float center = current[idx];
        float laplacian = current[idx - 1] + current[idx + 1] + current[idx - width] +
                          current[idx + width] - 4.0f * center;
        cpu[idx] = center + alpha * laplacian;
      }
    }

  float *d_current = nullptr, *d_next = nullptr;
  CHECK_CUDA(cudaMalloc(&d_current, current.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_next, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_current, current.data(), current.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  heat_step_kernel<<<blocks, threads>>>(d_current, d_next, width, height, alpha);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_next, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (std::size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-5f)
      ok = false;
  std::cout << "Center after step: " << gpu[(height / 2) * width + width / 2] << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_current));
  CHECK_CUDA(cudaFree(d_next));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
