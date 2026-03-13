// Example 089: Ray Sphere Tracer
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

__device__ __host__ Vec3 sub(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ __host__ float dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __host__ Vec3 normalize(Vec3 v) {
  float inv = 1.0f / sqrtf(dot(v, v));
  return {v.x * inv, v.y * inv, v.z * inv};
}

__device__ __host__ float trace_pixel(int x, int y, int width, int height) {
  float u = (2.0f * (x + 0.5f) / width - 1.0f);
  float v = (2.0f * (y + 0.5f) / height - 1.0f);
  Vec3 origin = {0.0f, 0.0f, -3.0f};
  Vec3 dir = normalize({u, -v, 1.5f});
  Vec3 center = {0.0f, 0.0f, 0.0f};
  Vec3 oc = sub(origin, center);
  float b = dot(oc, dir);
  float c = dot(oc, oc) - 1.0f;
  float discriminant = b * b - c;
  if (discriminant < 0.0f)
    return 0.0f;
  float t = -b - sqrtf(discriminant);
  if (t <= 0.0f)
    return 0.0f;
  Vec3 hit = {origin.x + t * dir.x, origin.y + t * dir.y, origin.z + t * dir.z};
  Vec3 normal = normalize(hit);
  Vec3 light = normalize({0.7f, 0.5f, -1.0f});
  return fmaxf(0.0f, dot(normal, light));
}

__global__ void ray_sphere_kernel(float *image, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height)
    image[y * width + x] = trace_pixel(x, y, width, height);
}

int main() {
  const int width = 16, height = 12;
  std::vector<float> cpu(width * height, 0.0f), gpu(width * height, 0.0f);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      cpu[y * width + x] = trace_pixel(x, y, width, height);

  float *d_image = nullptr;
  CHECK_CUDA(cudaMalloc(&d_image, gpu.size() * sizeof(float)));
  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  ray_sphere_kernel<<<blocks, threads>>>(d_image, width, height);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_image, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (std::size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-5f)
      ok = false;
  std::cout << "Center pixel shade: " << gpu[(height / 2) * width + width / 2] << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_image));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
