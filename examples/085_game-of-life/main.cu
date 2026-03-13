// Example 085: Game Of Life
// Track: Simulation
// Difficulty: Intermediate
// Status: Reference-friendly

#include <cuda_runtime.h>
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

__global__ void game_of_life_kernel(const int *current, int *next, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  int neighbors = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0)
        continue;
      int nx = (x + dx + width) % width;
      int ny = (y + dy + height) % height;
      neighbors += current[ny * width + nx];
    }
  int idx = y * width + x;
  int alive = current[idx];
  next[idx] = (neighbors == 3 || (alive && neighbors == 2)) ? 1 : 0;
}

int main() {
  const int width = 5, height = 5;
  std::vector<int> current(width * height, 0), cpu(width * height, 0), gpu(width * height, 0);
  current[1 * width + 2] = 1;
  current[2 * width + 2] = 1;
  current[3 * width + 2] = 1;
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      int neighbors = 0;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0)
            continue;
          int nx = (x + dx + width) % width;
          int ny = (y + dy + height) % height;
          neighbors += current[ny * width + nx];
        }
      int idx = y * width + x;
      int alive = current[idx];
      cpu[idx] = (neighbors == 3 || (alive && neighbors == 2)) ? 1 : 0;
    }

  int *d_current = nullptr, *d_next = nullptr;
  CHECK_CUDA(cudaMalloc(&d_current, current.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_next, gpu.size() * sizeof(int)));
  CHECK_CUDA(
      cudaMemcpy(d_current, current.data(), current.size() * sizeof(int), cudaMemcpyHostToDevice));
  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  game_of_life_kernel<<<blocks, threads>>>(d_current, d_next, width, height);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_next, gpu.size() * sizeof(int), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (std::size_t i = 0; i < gpu.size(); ++i)
    if (cpu[i] != gpu[i])
      ok = false;
  std::cout << "Center row after update:";
  for (int x = 0; x < width; ++x)
    std::cout << " " << gpu[2 * width + x];
  std::cout << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_current));
  CHECK_CUDA(cudaFree(d_next));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
