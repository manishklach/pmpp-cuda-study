#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "pmpp/benchmark.cuh"
#include "pmpp/cli.cuh"
#include "pmpp/cuda_check.cuh"
#include "pmpp/random_inputs.cuh"
#include "pmpp/report.cuh"
#include "pmpp/types.cuh"

namespace {

constexpr const char *kExampleName = "080_n-body-tiled";
constexpr float kSoftening = 1.0e-3f;
constexpr int kTileSize = 128;

struct Vec3 {
  float x;
  float y;
  float z;
};

pmpp::ValidationSummary compare_vec3_vectors(const std::vector<Vec3> &expected,
                                             const std::vector<Vec3> &actual, float tolerance) {
  pmpp::ValidationSummary summary{};
  if (expected.size() != actual.size()) {
    summary.ok = false;
    summary.mismatch_count = 1;
    return summary;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double dx = std::fabs(expected[i].x - actual[i].x);
    const double dy = std::fabs(expected[i].y - actual[i].y);
    const double dz = std::fabs(expected[i].z - actual[i].z);
    summary.max_abs_error = std::max(summary.max_abs_error, std::max(dx, std::max(dy, dz)));
    if (dx > tolerance || dy > tolerance || dz > tolerance) {
      summary.ok = false;
      ++summary.mismatch_count;
    }
  }
  return summary;
}

__global__ void nbody_tiled_kernel(const Vec3 *positions, Vec3 *accelerations, int n,
                                   float softening) {
  __shared__ Vec3 tile[kTileSize];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  Vec3 pi = positions[i];
  Vec3 acc = {0.0f, 0.0f, 0.0f};
  const int tiles = (n + blockDim.x - 1) / blockDim.x;

  for (int tile_idx = 0; tile_idx < tiles; ++tile_idx) {
    int j = tile_idx * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = j < n ? positions[j] : Vec3{0.0f, 0.0f, 0.0f};
    __syncthreads();

    int limit = min(blockDim.x, n - tile_idx * blockDim.x);
    for (int k = 0; k < limit; ++k) {
      int global_j = tile_idx * blockDim.x + k;
      if (global_j == i)
        continue;

      float dx = tile[k].x - pi.x;
      float dy = tile[k].y - pi.y;
      float dz = tile[k].z - pi.z;
      float dist2 = dx * dx + dy * dy + dz * dz + softening;
      float inv = rsqrtf(dist2);
      float inv3 = inv * inv * inv;
      acc.x += dx * inv3;
      acc.y += dy * inv3;
      acc.z += dz * inv3;
    }
    __syncthreads();
  }

  accelerations[i] = acc;
}

std::vector<Vec3> make_positions(int n, unsigned int seed) {
  std::vector<float> coords = pmpp::make_uniform_floats(n * 3, seed, -1.0f, 1.0f);
  std::vector<Vec3> positions(n);
  for (int i = 0; i < n; ++i)
    positions[i] = {coords[i * 3 + 0], coords[i * 3 + 1], coords[i * 3 + 2]};
  return positions;
}

std::vector<Vec3> cpu_reference(const std::vector<Vec3> &positions) {
  const int n = static_cast<int>(positions.size());
  std::vector<Vec3> output(n, {0.0f, 0.0f, 0.0f});
  for (int i = 0; i < n; ++i) {
    Vec3 acc = {0.0f, 0.0f, 0.0f};
    for (int j = 0; j < n; ++j) {
      if (i == j)
        continue;
      float dx = positions[j].x - positions[i].x;
      float dy = positions[j].y - positions[i].y;
      float dz = positions[j].z - positions[i].z;
      float dist2 = dx * dx + dy * dy + dz * dz + kSoftening;
      float inv = 1.0f / std::sqrt(dist2);
      float inv3 = inv * inv * inv;
      acc.x += dx * inv3;
      acc.y += dy * inv3;
      acc.z += dz * inv3;
    }
    output[i] = acc;
  }
  return output;
}

std::vector<Vec3> run_gpu_once(const std::vector<Vec3> &positions) {
  const int n = static_cast<int>(positions.size());
  std::vector<Vec3> output(n, {0.0f, 0.0f, 0.0f});

  Vec3 *device_positions = nullptr;
  Vec3 *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_positions, n * sizeof(Vec3)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(Vec3)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_positions, positions.data(), n * sizeof(Vec3),
                             cudaMemcpyHostToDevice));

  const int threads = kTileSize;
  const int blocks = (n + threads - 1) / threads;
  nbody_tiled_kernel<<<blocks, threads>>>(device_positions, device_output, n, kSoftening);
  PMPP_CUDA_KERNEL_CHECK();

  PMPP_CUDA_CHECK(cudaMemcpy(output.data(), device_output, n * sizeof(Vec3), cudaMemcpyDeviceToHost));
  PMPP_CUDA_CHECK(cudaFree(device_positions));
  PMPP_CUDA_CHECK(cudaFree(device_output));
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  std::vector<Vec3> positions = make_positions(options.size, options.seed);
  std::vector<Vec3> cpu = cpu_reference(positions);
  std::vector<Vec3> gpu = run_gpu_once(positions);

  pmpp::ValidationSummary summary = compare_vec3_vectors(cpu, gpu, 1.0e-3f);
  summary.notes = "This tiled N-body variant reuses particle positions through shared memory.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int n = options.size;
  std::vector<Vec3> positions = make_positions(n, options.seed);
  Vec3 *device_positions = nullptr;
  Vec3 *device_output = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_positions, n * sizeof(Vec3)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(Vec3)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_positions, positions.data(), n * sizeof(Vec3),
                             cudaMemcpyHostToDevice));

  const int threads = kTileSize;
  const int blocks = (n + threads - 1) / threads;
  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    nbody_tiled_kernel<<<blocks, threads>>>(device_positions, device_output, n, kSoftening);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps(static_cast<std::size_t>(n) * sizeof(Vec3) * 2,
                                              stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(n, stats.avg_ms);

  PMPP_CUDA_CHECK(cudaFree(device_positions));
  PMPP_CUDA_CHECK(cudaFree(device_output));
  return stats;
}

}  // namespace

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);
  if (options.size < 2)
    options.size = 2;

  if (options.check) {
    pmpp::ValidationSummary summary = run_check(options);
    pmpp::print_validation_report(kExampleName, summary);
    if (!summary.ok)
      return EXIT_FAILURE;
  }

  if (options.bench) {
    if (!options.verify)
      std::cout << "Validation: skipped (benchmark mode, use --verify or add --check)." << std::endl;
    pmpp::BenchmarkStats stats = run_bench(options);
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters,
                                 "Particles/sec");
  }

  return EXIT_SUCCESS;
}
