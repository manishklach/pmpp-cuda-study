// Example 084: Lattice Boltzmann Step
// Track: Simulation
// Difficulty: Advanced
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

constexpr int kDirs = 9;
__constant__ int kDx[kDirs] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__constant__ int kDy[kDirs] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
__constant__ float kW[kDirs] = {4.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f, 1.0f / 9.0f,
                                1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};

__global__ void lbm_step_kernel(const float *current, float *next, int width, int height,
                                float omega) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  int cell = y * width + x;
  float rho = 0.0f;
  for (int d = 0; d < kDirs; ++d)
    rho += current[d * width * height + cell];
  for (int d = 0; d < kDirs; ++d) {
    float feq = kW[d] * rho;
    float post =
        current[d * width * height + cell] + omega * (feq - current[d * width * height + cell]);
    int nx = (x + kDx[d] + width) % width;
    int ny = (y + kDy[d] + height) % height;
    int dest = ny * width + nx;
    next[d * width * height + dest] = post;
  }
}

int main() {
  const int width = 4, height = 4, cells = width * height;
  const float omega = 0.8f;
  const int host_dx[kDirs] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
  const int host_dy[kDirs] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
  std::vector<float> current(kDirs * cells, 0.0f), cpu(kDirs * cells, 0.0f),
      gpu(kDirs * cells, 0.0f);
  for (int d = 0; d < kDirs; ++d)
    for (int cell = 0; cell < cells; ++cell)
      current[d * cells + cell] = kW[d] * (1.0f + 0.01f * cell);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      int cell = y * width + x;
      float rho = 0.0f;
      for (int d = 0; d < kDirs; ++d)
        rho += current[d * cells + cell];
      for (int d = 0; d < kDirs; ++d) {
        float feq = kW[d] * rho;
        float post = current[d * cells + cell] + omega * (feq - current[d * cells + cell]);
        int nx = (x + host_dx[d] + width) % width;
        int ny = (y + host_dy[d] + height) % height;
        cpu[d * cells + ny * width + nx] = post;
      }
    }

  float *d_current = nullptr, *d_next = nullptr;
  CHECK_CUDA(cudaMalloc(&d_current, current.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_next, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_current, current.data(), current.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_next, 0, gpu.size() * sizeof(float)));
  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  lbm_step_kernel<<<blocks, threads>>>(d_current, d_next, width, height, omega);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_next, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (std::size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-5f)
      ok = false;
  std::cout << "Population sample: " << gpu[0] << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_current));
  CHECK_CUDA(cudaFree(d_next));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
