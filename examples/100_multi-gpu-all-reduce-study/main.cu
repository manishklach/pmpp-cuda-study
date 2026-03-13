// Example 100: Multi GPU All Reduce Study
// Track: Graph and ML
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

__global__ void all_reduce_sum_kernel(const float *shard_a, const float *shard_b, float *reduced,
                                      int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    reduced[idx] = shard_a[idx] + shard_b[idx];
}

__global__ void broadcast_kernel(const float *reduced, float *out_a, float *out_b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out_a[idx] = reduced[idx];
    out_b[idx] = reduced[idx];
  }
}

int main() {
  const int n = 8;
  std::vector<float> shard_a = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> shard_b = {8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<float> cpu(n, 0.0f), gpu_reduce(n, 0.0f), gpu_a(n, 0.0f), gpu_b(n, 0.0f);
  for (int i = 0; i < n; ++i)
    cpu[i] = shard_a[i] + shard_b[i];

  float *d_a = nullptr, *d_b = nullptr, *d_reduce = nullptr, *d_out_a = nullptr, *d_out_b = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_reduce, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out_b, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, shard_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, shard_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  all_reduce_sum_kernel<<<1, 256>>>(d_a, d_b, d_reduce, n);
  broadcast_kernel<<<1, 256>>>(d_reduce, d_out_a, d_out_b, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu_reduce.data(), d_reduce, n * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(gpu_a.data(), d_out_a, n * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(gpu_b.data(), d_out_b, n * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (std::fabs(cpu[i] - gpu_reduce[i]) > 1.0e-6f || std::fabs(cpu[i] - gpu_a[i]) > 1.0e-6f ||
        std::fabs(cpu[i] - gpu_b[i]) > 1.0e-6f)
      ok = false;
  std::cout << "Reduced vector:";
  for (float value : gpu_reduce)
    std::cout << " " << value;
  std::cout << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_reduce));
  CHECK_CUDA(cudaFree(d_out_a));
  CHECK_CUDA(cudaFree(d_out_b));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
