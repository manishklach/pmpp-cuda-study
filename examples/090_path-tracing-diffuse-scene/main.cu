// Example 090: Path Tracing Diffuse Scene
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

struct Vec3 {
  float x;
  float y;
  float z;
};

__device__ __host__ float dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __host__ Vec3 normalize(Vec3 v) {
  float inv = 1.0f / sqrtf(dot(v, v));
  return {v.x * inv, v.y * inv, v.z * inv};
}
__device__ __host__ unsigned int lcg(unsigned int &state) {
  state = 1664525u * state + 1013904223u;
  return state;
}
__device__ __host__ float uniform01(unsigned int &state) {
  return (lcg(state) & 0x00FFFFFF) / static_cast<float>(0x01000000);
}

__device__ __host__ float estimate_pixel(int x, int y, int width, int height) {
  unsigned int state = 1234u + 97u * static_cast<unsigned int>(y * width + x);
  float u = (2.0f * (x + 0.5f) / width - 1.0f);
  float v = (2.0f * (y + 0.5f) / height - 1.0f);
  Vec3 origin = {0.0f, 0.0f, -3.0f};
  Vec3 dir = normalize({u, -v, 1.5f});
  float b = origin.x * dir.x + origin.y * dir.y + origin.z * dir.z;
  float c = origin.x * origin.x + origin.y * origin.y + origin.z * origin.z - 1.0f;
  float disc = b * b - c;
  if (disc < 0.0f)
    return 0.2f + 0.8f * fmaxf(0.0f, dir.y);
  float t = -b - sqrtf(disc);
  if (t <= 0.0f)
    return 0.2f + 0.8f * fmaxf(0.0f, dir.y);
  Vec3 hit = {origin.x + t * dir.x, origin.y + t * dir.y, origin.z + t * dir.z};
  Vec3 normal = normalize(hit);
  Vec3 random_dir =
      normalize({normal.x + 2.0f * uniform01(state) - 1.0f, normal.y + 2.0f * uniform01(state),
                 normal.z + 2.0f * uniform01(state) - 1.0f});
  float bounce = fmaxf(0.0f, random_dir.y);
  return 0.6f * bounce + 0.1f;
}

__global__ void path_trace_kernel(float *image, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height)
    image[y * width + x] = estimate_pixel(x, y, width, height);
}

int main() {
  const int width = 16, height = 12;
  std::vector<float> cpu(width * height, 0.0f), gpu(width * height, 0.0f);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      cpu[y * width + x] = estimate_pixel(x, y, width, height);

  float *d_image = nullptr;
  CHECK_CUDA(cudaMalloc(&d_image, gpu.size() * sizeof(float)));
  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  path_trace_kernel<<<blocks, threads>>>(d_image, width, height);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_image, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (std::size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-5f)
      ok = false;
  std::cout << "Center pixel radiance: " << gpu[(height / 2) * width + width / 2] << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_image));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
