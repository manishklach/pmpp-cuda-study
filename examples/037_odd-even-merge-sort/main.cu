// Example 037: Odd Even Merge Sort

// Track: Parallel Patterns
// Difficulty: Intermediate
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#define CHECK_CUDA(call)                                                                           \
  do {                                                                                             \
    cudaError_t status__ = (call);                                                                 \
    if (status__ != cudaSuccess) {                                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(status__) << " at " << __FILE__ << ":"     \
                << __LINE__ << std::endl;                                                          \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

__global__ void compare_swap_pairs_kernel(int *data, const int *left, const int *right,
                                          int pair_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < pair_count) {
    int a = left[idx], b = right[idx];
    if (data[a] > data[b]) {
      int t = data[a];
      data[a] = data[b];
      data[b] = t;
    }
  }
}
int main() {
  const int n = 16;
  std::vector<int> input = {15, 3, 14, 2, 13, 1, 12, 0, 11, 7, 10, 6, 9, 5, 8, 4};
  auto cpu = input;
  std::sort(cpu.begin(), cpu.end());
  std::vector<std::pair<int, int>> pairs;
  for (int p = 1; p < n; p *= 2) {
    for (int i = 0; i < n; i += 2 * p) {
      for (int j = 0; j < p && i + j + p < n; ++j) {
        pairs.push_back({i + j, i + j + p});
      }
    }
  }
  std::vector<int> left(pairs.size()), right(pairs.size());
  for (size_t i = 0; i < pairs.size(); ++i) {
    left[i] = pairs[i].first;
    right[i] = pairs[i].second;
  }
  int *d = nullptr, *dl = nullptr, *dr = nullptr;
  CHECK_CUDA(cudaMalloc(&d, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dl, left.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dr, right.size() * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dl, left.data(), left.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dr, right.data(), right.size() * sizeof(int), cudaMemcpyHostToDevice));
  for (int pass = 0; pass < n; ++pass) {
    compare_swap_pairs_kernel<<<1, 128>>>(d, dl, dr, (int)left.size());
    CHECK_CUDA(cudaGetLastError());
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<int> gpu(n);
  CHECK_CUDA(cudaMemcpy(gpu.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
  bool ok = gpu == cpu;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d));
  CHECK_CUDA(cudaFree(dl));
  CHECK_CUDA(cudaFree(dr));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
