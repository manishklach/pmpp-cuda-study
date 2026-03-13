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

__global__ void example_kernel(const float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    output[idx] = input[idx];
}

int main() {
  const int n = 1024;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  std::vector<float> host_input(n, 0.0f);
  std::vector<float> host_output(n, 0.0f);
  std::vector<float> host_reference(n, 0.0f);

  for (int i = 0; i < n; ++i) {
    host_input[i] = static_cast<float>(i);
    host_reference[i] = host_input[i];
  }

  float *device_input = nullptr;
  float *device_output = nullptr;
  CHECK_CUDA(cudaMalloc(&device_input, bytes));
  CHECK_CUDA(cudaMalloc(&device_output, bytes));

  CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  example_kernel<<<blocks, threads>>>(device_input, device_output, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    if (std::fabs(host_output[i] - host_reference[i]) > 1.0e-6f) {
      ok = false;
      break;
    }
  }

  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;

  CHECK_CUDA(cudaFree(device_input));
  CHECK_CUDA(cudaFree(device_output));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
