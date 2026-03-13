// Example 039: Merge Two Sorted Arrays

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

__global__ void merge_kernel(const int *a, int na, const int *b, int nb, int *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = na + nb;
  if (idx < total) {
    int lo = max(0, idx - nb);
    int hi = min(idx, na);
    while (lo < hi) {
      int mid = (lo + hi + 1) / 2;
      if (a[mid - 1] > b[idx - mid])
        hi = mid - 1;
      else
        lo = mid;
    }
    int i = lo;
    int j = idx - i;
    int a_val = i < na ? a[i] : INT_MAX;
    int b_val = j < nb ? b[j] : INT_MAX;
    out[idx] = min(a_val, b_val);
  }
}
int main() {
  std::vector<int> a = {1, 4, 7, 10, 13, 16, 19, 22};
  std::vector<int> b = {0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17};
  std::vector<int> cpu(a.size() + b.size()), gpu(cpu.size());
  std::merge(a.begin(), a.end(), b.begin(), b.end(), cpu.begin());
  int *da = nullptr, *db = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&da, a.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&db, b.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&do_, cpu.size() * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(da, a.data(), a.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b.data(), b.size() * sizeof(int), cudaMemcpyHostToDevice));
  merge_kernel<<<1, 128>>>(da, (int)a.size(), db, (int)b.size(), do_);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), do_, cpu.size() * sizeof(int), cudaMemcpyDeviceToHost));
  bool ok = gpu == cpu;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(do_));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
