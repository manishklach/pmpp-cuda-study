// Example 073: Run Length Encoding
// Track: Image and Signal
// Difficulty: Advanced
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

__global__ void rle_kernel(const int *input, int n, int *values, int *lengths, int *run_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
    int runs = 0;
    for (int i = 0; i < n; ++i)
      if (i == 0 || input[i] != input[i - 1])
        ++runs;
    *run_count = runs;
  }
  __syncthreads();

  if (idx >= n)
    return;
  bool is_run_start = (idx == 0) || (input[idx] != input[idx - 1]);
  if (!is_run_start)
    return;

  int length = 1;
  while (idx + length < n && input[idx + length] == input[idx])
    ++length;

  int slot = 0;
  for (int i = 0; i < idx; ++i)
    if (i == 0 || input[i] != input[i - 1])
      ++slot;

  values[slot] = input[idx];
  lengths[slot] = length;
}

int main() {
  std::vector<int> input = {1, 1, 1, 2, 2, 5, 5, 5, 5, 3, 3, 1, 1, 4};
  const int n = static_cast<int>(input.size());
  std::vector<int> cpu_values, cpu_lengths;
  for (int i = 0; i < n;) {
    int j = i + 1;
    while (j < n && input[j] == input[i])
      ++j;
    cpu_values.push_back(input[i]);
    cpu_lengths.push_back(j - i);
    i = j;
  }

  std::vector<int> gpu_values(n, 0), gpu_lengths(n, 0);
  int gpu_runs = 0;
  int *d_input = nullptr, *d_values = nullptr, *d_lengths = nullptr, *d_run_count = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_values, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_lengths, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_run_count, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_run_count, 0, sizeof(int)));
  rle_kernel<<<1, 128>>>(d_input, n, d_values, d_lengths, d_run_count);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu_values.data(), d_values, n * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(gpu_lengths.data(), d_lengths, n * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&gpu_runs, d_run_count, sizeof(int), cudaMemcpyDeviceToHost));

  bool ok = gpu_runs == static_cast<int>(cpu_values.size());
  for (int i = 0; i < gpu_runs && ok; ++i)
    if (gpu_values[i] != cpu_values[i] || gpu_lengths[i] != cpu_lengths[i])
      ok = false;
  std::cout << "Run count: " << gpu_runs << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_values));
  CHECK_CUDA(cudaFree(d_lengths));
  CHECK_CUDA(cudaFree(d_run_count));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
