// Example 052: Sparse Matrix Vector Multiply CSR
// Difficulty: Advanced

// Track: Linear Algebra
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#define CHECK_CUDA(call)                                                                       \
  do {                                                                                         \
    cudaError_t status__ = (call);                                                             \
    if (status__ != cudaSuccess) {                                                             \
      std::cerr << "CUDA error: " << cudaGetErrorString(status__) << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;                                                      \
      std::exit(EXIT_FAILURE);                                                                 \
    }                                                                                          \
  } while (0)

__global__ void spmv_csr_kernel(const int *row_ptr, const int *col_idx, const float *values,
                                const float *x, float *y, int rows) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int jj = row_ptr[row]; jj < row_ptr[row + 1]; ++jj)
      sum += values[jj] * x[col_idx[jj]];
    y[row] = sum;
  }
}
int main() {
  const int rows = 4, cols = 4;
  std::vector<int> row_ptr = {0, 2, 4, 7, 8}, col_idx = {0, 2, 1, 3, 0, 2, 3, 1};
  std::vector<float> vals = {10, 2, 3, 9, 7, 8, 7, 5}, x = {1, 2, 3, 4}, gpu(rows, 0.0f),
                     cpu(rows, 0.0f);
  for (int r = 0; r < rows; ++r)
    for (int jj = row_ptr[r]; jj < row_ptr[r + 1]; ++jj)
      cpu[r] += vals[jj] * x[col_idx[jj]];
  int *dr = nullptr, *dc = nullptr;
  float *dv = nullptr, *dx = nullptr, *dy = nullptr;
  CHECK_CUDA(cudaMalloc(&dr, row_ptr.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dc, col_idx.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dv, vals.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dx, x.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dy, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dr, row_ptr.data(), row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dc, col_idx.data(), col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dv, vals.data(), vals.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
  spmv_csr_kernel<<<1, 64>>>(dr, dc, dv, dx, dy, rows);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), dy, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < rows; ++i)
    if (std::fabs(gpu[i] - cpu[i]) > 1e-5f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(dr));
  CHECK_CUDA(cudaFree(dc));
  CHECK_CUDA(cudaFree(dv));
  CHECK_CUDA(cudaFree(dx));
  CHECK_CUDA(cudaFree(dy));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
