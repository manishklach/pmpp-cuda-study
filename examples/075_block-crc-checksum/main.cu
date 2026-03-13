// Example 075: Block CRC Checksum
// Track: Image and Signal
// Difficulty: Advanced
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <cstdint>
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

__device__ __host__ uint32_t crc32_update(uint32_t crc, unsigned char byte) {
  crc ^= byte;
  for (int i = 0; i < 8; ++i)
    crc = (crc >> 1) ^ (0xEDB88320u & (-(static_cast<int>(crc & 1u))));
  return crc;
}

__global__ void block_crc_kernel(const unsigned char *data, uint32_t *crc_out, int total_bytes,
                                 int chunk_size) {
  int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  int start = chunk * chunk_size;
  if (start >= total_bytes)
    return;
  int end = min(start + chunk_size, total_bytes);
  uint32_t crc = 0xFFFFFFFFu;
  for (int i = start; i < end; ++i)
    crc = crc32_update(crc, data[i]);
  crc_out[chunk] = crc ^ 0xFFFFFFFFu;
}

int main() {
  const int total_bytes = 64, chunk_size = 16;
  const int chunks = (total_bytes + chunk_size - 1) / chunk_size;
  std::vector<unsigned char> data(total_bytes);
  std::vector<uint32_t> cpu(chunks, 0u), gpu(chunks, 0u);
  for (int i = 0; i < total_bytes; ++i)
    data[i] = static_cast<unsigned char>((13 * i + 7) & 0xFF);
  for (int chunk = 0; chunk < chunks; ++chunk) {
    int start = chunk * chunk_size;
    int end = std::min(start + chunk_size, total_bytes);
    uint32_t crc = 0xFFFFFFFFu;
    for (int i = start; i < end; ++i)
      crc = crc32_update(crc, data[i]);
    cpu[chunk] = crc ^ 0xFFFFFFFFu;
  }

  unsigned char *d_data = nullptr;
  uint32_t *d_crc = nullptr;
  CHECK_CUDA(cudaMalloc(&d_data, total_bytes * sizeof(unsigned char)));
  CHECK_CUDA(cudaMalloc(&d_crc, chunks * sizeof(uint32_t)));
  CHECK_CUDA(
      cudaMemcpy(d_data, data.data(), total_bytes * sizeof(unsigned char), cudaMemcpyHostToDevice));
  block_crc_kernel<<<1, 128>>>(d_data, d_crc, total_bytes, chunk_size);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(gpu.data(), d_crc, chunks * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < chunks; ++i)
    if (cpu[i] != gpu[i])
      ok = false;
  std::cout << "First CRC: " << gpu[0] << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_data));
  CHECK_CUDA(cudaFree(d_crc));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
