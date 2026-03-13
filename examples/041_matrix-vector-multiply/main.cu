// Example 041: Matrix Vector Multiply
// Difficulty: Intermediate

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

__global__ void matvec_kernel(const float *matrix, const float *vector, float *output, int rows,
                              int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col)
      sum += matrix[row * cols + col] * vector[col];
    output[row] = sum;
  }
}
int main() {
  const int rows = 16, cols = 8;
  std::vector<float> matrix(rows * cols), vector(cols), gpu(rows, 0.0f), cpu(rows, 0.0f);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      matrix[r * cols + c] = static_cast<float>((r + c) % 7 + 1);
  for (int c = 0; c < cols; ++c)
    vector[c] = static_cast<float>(c + 1);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      cpu[r] += matrix[r * cols + c] * vector[c];
  float *dm = nullptr, *dv = nullptr, *do_ = nullptr;
  CHECK_CUDA(cudaMalloc(&dm, matrix.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dv, vector.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&do_, gpu.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dm, matrix.data(), matrix.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dv, vector.data(), vector.size() * sizeof(float), cudaMemcpyHostToDevice));
  matvec_kernel<<<1, 64>>>(dm, dv, do_, rows, cols);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), do_, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < rows; ++i)
    if (std::fabs(gpu[i] - cpu[i]) > 1e-5f)
      ok = false;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(dm));
  CHECK_CUDA(cudaFree(dv));
  CHECK_CUDA(cudaFree(do_));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
