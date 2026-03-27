#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include "pmpp/benchmark.cuh"
#include "pmpp/cli.cuh"
#include "pmpp/compare.cuh"
#include "pmpp/cuda_check.cuh"
#include "pmpp/random_inputs.cuh"
#include "pmpp/report.cuh"

namespace {

constexpr const char *kExampleName = "101_segmented-reduction";
constexpr int kBlockSize = 128;
constexpr int kSegmentLength = 64;

__global__ void segmented_reduce_kernel(const float *values, const int *offsets, float *segment_sums,
                                        int segment_count) {
  __shared__ float scratch[kBlockSize];
  int segment = blockIdx.x;
  if (segment >= segment_count)
    return;

  int begin = offsets[segment];
  int end = offsets[segment + 1];
  float local_sum = 0.0f;

  // One block owns one contiguous segment. Threads walk the segment with a simple stride so the
  // intra-segment work sharing is easy to inspect before tackling irregular segment lengths.
  for (int index = begin + threadIdx.x; index < end; index += blockDim.x)
    local_sum += values[index];

  scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0)
    segment_sums[segment] = scratch[0];
}

std::vector<float> cpu_reference(const std::vector<float> &values, const std::vector<int> &offsets) {
  std::vector<float> output(offsets.size() - 1, 0.0f);
  for (std::size_t segment = 0; segment + 1 < offsets.size(); ++segment)
    for (int i = offsets[segment]; i < offsets[segment + 1]; ++i)
      output[segment] += values[i];
  return output;
}

pmpp::ValidationSummary run_check(const pmpp::CommonOptions &options) {
  const int segment_count = std::max(4, options.size / kSegmentLength);
  const int element_count = segment_count * kSegmentLength;
  std::vector<float> values = pmpp::make_uniform_floats(element_count, options.seed, -2.0f, 2.0f);
  std::vector<int> offsets(segment_count + 1, 0);
  for (int segment = 0; segment <= segment_count; ++segment)
    offsets[segment] = segment * kSegmentLength;

  std::vector<float> cpu = cpu_reference(values, offsets);
  std::vector<float> gpu(segment_count, 0.0f);

  float *device_values = nullptr;
  float *device_output = nullptr;
  int *device_offsets = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_values, element_count * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_offsets, offsets.size() * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, segment_count * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_values, values.data(), element_count * sizeof(float), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice));

  segmented_reduce_kernel<<<segment_count, kBlockSize>>>(device_values, device_offsets, device_output,
                                                         segment_count);
  PMPP_CUDA_KERNEL_CHECK();
  PMPP_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output, segment_count * sizeof(float), cudaMemcpyDeviceToHost));

  PMPP_CUDA_CHECK(cudaFree(device_values));
  PMPP_CUDA_CHECK(cudaFree(device_offsets));
  PMPP_CUDA_CHECK(cudaFree(device_output));

  pmpp::ValidationSummary summary = pmpp::compare_vectors(cpu, gpu, 1.0e-5f);
  summary.notes = "Each block reduces one contiguous segment, which is a good first segmented-reduction baseline.";
  return summary;
}

pmpp::BenchmarkStats run_bench(const pmpp::CommonOptions &options) {
  const int segment_count = std::max(4, options.size / kSegmentLength);
  const int element_count = segment_count * kSegmentLength;
  std::vector<float> values = pmpp::make_uniform_floats(element_count, options.seed, -2.0f, 2.0f);
  std::vector<int> offsets(segment_count + 1, 0);
  for (int segment = 0; segment <= segment_count; ++segment)
    offsets[segment] = segment * kSegmentLength;

  float *device_values = nullptr;
  float *device_output = nullptr;
  int *device_offsets = nullptr;
  PMPP_CUDA_CHECK(cudaMalloc(&device_values, element_count * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_offsets, offsets.size() * sizeof(int)));
  PMPP_CUDA_CHECK(cudaMalloc(&device_output, segment_count * sizeof(float)));
  PMPP_CUDA_CHECK(cudaMemcpy(device_values, values.data(), element_count * sizeof(float), cudaMemcpyHostToDevice));
  PMPP_CUDA_CHECK(cudaMemcpy(device_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice));

  pmpp::BenchmarkStats stats = pmpp::run_benchmark_loop(options.warmup, options.iters, [&] {
    segmented_reduce_kernel<<<segment_count, kBlockSize>>>(device_values, device_offsets, device_output,
                                                           segment_count);
    PMPP_CUDA_KERNEL_CHECK();
  });
  stats.bandwidth_gbps = pmpp::bandwidth_gbps((static_cast<std::size_t>(element_count) + offsets.size()) * sizeof(float), stats.avg_ms);
  stats.throughput = pmpp::elements_per_second(element_count, stats.avg_ms);
  stats.problem_label = "Input elements";
  stats.problem_size = static_cast<std::size_t>(element_count);

  PMPP_CUDA_CHECK(cudaFree(device_values));
  PMPP_CUDA_CHECK(cudaFree(device_offsets));
  PMPP_CUDA_CHECK(cudaFree(device_output));
  return stats;
}

}  // namespace

int main(int argc, char **argv) {
  pmpp::CommonOptions options = pmpp::parse_common_options(argc, argv);

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
    pmpp::print_benchmark_report(kExampleName, stats, options.warmup, options.iters, "Elements/sec");
  }

  return EXIT_SUCCESS;
}
