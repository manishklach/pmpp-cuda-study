// Example 072: Delta Encoding
// Track: Image and Signal
// Difficulty: Beginner
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

__global__ void delta_encode_kernel(const int *input, int *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    output[idx] = idx == 0 ? input[idx] : input[idx] - input[idx - 1];
}

int main() {
  std::vector<int> input = {3, 7, 10, 18, 19, 27, 30, 44, 50, 63};
  const int n = static_cast<int>(input.size());
  std::vector<int> cpu(n, 0), gpu(n, 0);
  for (int i = 0; i < n; ++i)
    cpu[i] = i == 0 ? input[i] : input[i] - input[i - 1];

  int *d_input = nullptr, *d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  delta_encode_kernel<<<1, 128>>>(d_input, d_output, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (cpu[i] != gpu[i])
      ok = false;
  std::cout << "Encoded stream:";
  for (int value : gpu)
    std::cout << " " << value;
  std::cout << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
