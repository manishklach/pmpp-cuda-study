// Example 074: Parallel Base64 Or Hex Encode
// Track: Image and Signal
// Difficulty: Intermediate
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <string>

inline void check_cuda(cudaError_t status, const char *file, int line) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << file << ":" << line
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

__constant__ char kHexDigits[16] = {'0', '1', '2', '3', '4', '5', '6', '7',
                                    '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

__global__ void hex_encode_kernel(const unsigned char *input, char *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    unsigned char value = input[idx];
    output[2 * idx] = kHexDigits[value >> 4];
    output[2 * idx + 1] = kHexDigits[value & 0x0F];
  }
}

int main() {
  std::string text = "PMPP CUDA";
  const int n = static_cast<int>(text.size());
  std::string cpu(2 * n, '0'), gpu(2 * n, '0');
  const char *hex = "0123456789ABCDEF";
  for (int i = 0; i < n; ++i) {
    unsigned char value = static_cast<unsigned char>(text[i]);
    cpu[2 * i] = hex[value >> 4];
    cpu[2 * i + 1] = hex[value & 0x0F];
  }

  unsigned char *d_input = nullptr;
  char *d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(unsigned char)));
  CHECK_CUDA(cudaMalloc(&d_output, 2 * n * sizeof(char)));
  CHECK_CUDA(cudaMemcpy(d_input, text.data(), n * sizeof(unsigned char), cudaMemcpyHostToDevice));
  hex_encode_kernel<<<1, 128>>>(d_input, d_output, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_output, 2 * n * sizeof(char), cudaMemcpyDeviceToHost));

  bool ok = cpu == gpu;
  std::cout << "Encoded text: " << gpu << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
