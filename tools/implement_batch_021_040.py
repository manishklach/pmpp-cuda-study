from pathlib import Path
from textwrap import dedent

ROOT = Path(r"C:\Users\ManishKL\Documents\Playground\pmpp-cuda-study")
EX = ROOT / "examples"

DATA = {
21: ("021_dot-product", "Dot Product", ["map-reduce structure", "block-level reduction", "multi-pass accumulation"], ["Increase vector size and inspect block count.", "Swap to double precision for the CPU reference."]),
22: ("022_l2-norm", "L2 Norm", ["sum of squares", "reduction reuse", "host-side finalization"], ["Compare norm and squared norm.", "Try vectors with larger dynamic range."]),
23: ("023_sum-reduction", "Sum Reduction", ["shared-memory reduction", "block partials", "final host aggregation"], ["Measure different block sizes.", "Add a second GPU reduction pass."]),
24: ("024_max-reduction", "Max Reduction", ["identity values", "parallel max", "partial reductions"], ["Use random inputs with a known maximum.", "Compare against Thrust later."]),
25: ("025_min-reduction", "Min Reduction", ["parallel min", "sentinel initialization", "result validation"], ["Mix positive and negative values.", "Add location tracking for the min index."]),
26: ("026_prefix-sum-naive-scan", "Prefix Sum Naive Scan", ["Hillis-Steele scan", "iterative passes", "inclusive prefix behavior"], ["Convert to exclusive scan.", "Handle larger-than-one-block inputs."]),
27: ("027_prefix-sum-work-efficient-scan", "Prefix Sum Work Efficient Scan", ["Blelloch upsweep/downsweep", "shared memory", "work efficiency"], ["Turn the inclusive result into exclusive form.", "Tile across multiple blocks."]),
28: ("028_histogram-global-atomics", "Histogram Global Atomics", ["global atomics", "bin contention", "input distributions"], ["Vary the number of bins.", "Compare against the shared-memory version."]),
29: ("029_histogram-shared-memory", "Histogram Shared Memory", ["block-private bins", "shared-memory aggregation", "global merge"], ["Increase bin count.", "Stress-test skewed input distributions."]),
30: ("030_stream-compaction", "Stream Compaction", ["predicate filtering", "atomic output reservation", "packed outputs"], ["Swap the predicate.", "Replace atomics with scan-based compaction later."]),
31: ("031_gather", "Gather", ["indirect reads", "index arrays", "memory locality"], ["Try repeated indices.", "Compare sorted versus random index patterns."]),
32: ("032_scatter", "Scatter", ["indirect writes", "permutation safety", "destination mapping"], ["Deliberately create collisions and reason about them.", "Use a permutation inverse to validate."]),
33: ("033_predicate-count", "Predicate Count", ["boolean predicates", "atomic counts", "selectivity"], ["Count positives, evens, or range matches.", "Compare atomic count with reduction-based counting."]),
34: ("034_find-first-match", "Find First Match", ["atomicMin", "sentinel initialization", "first-hit semantics"], ["Search for a missing value.", "Track both first and last match."]),
35: ("035_parallel-even-odd-sort", "Parallel Even Odd Sort", ["alternating compare-swap phases", "small-array sorting", "iterative kernel launches"], ["Increase the array size gradually.", "Compare with bitonic sort on the same data."]),
36: ("036_bitonic-sort", "Bitonic Sort", ["sorting networks", "structured compare-exchange", "power-of-two inputs"], ["Sort descending instead of ascending.", "Use random seeds for stronger tests."]),
37: ("037_odd-even-merge-sort", "Odd Even Merge Sort", ["merge network structure", "compare-exchange pairs", "network-style sorting"], ["Visualize the stages on paper.", "Compare its output and cost with bitonic sort."]),
38: ("038_parallel-binary-search-over-sorted-chunks", "Parallel Binary Search Over Sorted Chunks", ["batched queries", "independent searches", "chunk boundaries"], ["Use different chunk sizes.", "Return insertion positions for missing values."]),
39: ("039_merge-two-sorted-arrays", "Merge Two Sorted Arrays", ["merge path intuition", "parallel merge positions", "binary partitioning"], ["Try uneven input lengths.", "Preserve stability on equal elements."]),
40: ("040_top-k-selection", "Top K Selection", ["selection versus full sort", "partial ordering", "GPU-assisted ranking"], ["Change K and inspect output stability.", "Replace full sort with iterative selection later."]),
}

TRACK = "Parallel Patterns"


def w(path: Path, text: str):
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")


def readme(i: int) -> str:
    _, title, focus, modify = DATA[i]
    lines = [
        f"# {i:03d} - {title}", "",
        f"- Track: `{TRACK}`", "- Difficulty: `Intermediate`", "- Status: `Reference-friendly`", "- GitHub batch: `021-040`", "",
        "## Goal", "", f"Build and study a working CUDA implementation of **{title}**.", "",
        "## PMPP Ideas To Focus On", "",
    ]
    lines.extend(f"- {x}" for x in focus)
    lines.extend([
        "", "## Build", "", "```powershell", "nvcc -std=c++17 -O2 main.cu -o example.exe", "```", "",
        "## Run", "", "```powershell", ".\\example.exe", "```", "",
        "## Validation", "", "- The program prints `PASS` when GPU output matches the CPU reference.", "- These examples use intentionally small inputs so each pattern is easy to inspect first.", "",
        "## What To Modify Next", "",
    ])
    lines.extend(f"- {x}" for x in modify)
    return "\n".join(lines)

COMMON = """
// Track: Parallel Patterns
// Difficulty: Intermediate
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
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
"""


def head(i: int, title: str) -> str:
    return f"// Example {i:03d}: {title}\n" + COMMON


def simple_template(i: int, title: str, body: str) -> str:
    return head(i, title) + dedent(body)
from textwrap import dedent

CODE = {}

CODE[21] = simple_template(21, DATA[21][1], r'''
__global__ void dot_partials_kernel(const float* a, const float* b, float* partials, int n) {
  __shared__ float scratch[256];
  int global = blockIdx.x * blockDim.x + threadIdx.x;
  int local = threadIdx.x;
  float value = global < n ? a[global] * b[global] : 0.0f;
  scratch[local] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local < stride) scratch[local] += scratch[local + stride];
    __syncthreads();
  }
  if (local == 0) partials[blockIdx.x] = scratch[0];
}

int main() {
  const int n = 1 << 12;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  std::vector<float> a(n), b(n), partials(blocks, 0.0f);
  float cpu = 0.0f;
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>((i % 17) - 8) * 0.25f;
    b[i] = static_cast<float>((i % 11) - 5) * 0.5f;
    cpu += a[i] * b[i];
  }
  float *da = nullptr, *db = nullptr, *dp = nullptr;
  CHECK_CUDA(cudaMalloc(&da, bytes));
  CHECK_CUDA(cudaMalloc(&db, bytes));
  CHECK_CUDA(cudaMalloc(&dp, blocks * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
  dot_partials_kernel<<<blocks, threads>>>(da, db, dp, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(partials.data(), dp, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  float gpu = std::accumulate(partials.begin(), partials.end(), 0.0f);
  std::cout << "CPU dot: " << cpu << "\nGPU dot: " << gpu << std::endl;
  std::cout << "Validation: " << (std::fabs(cpu - gpu) < 1.0e-3f ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(da)); CHECK_CUDA(cudaFree(db)); CHECK_CUDA(cudaFree(dp));
  return std::fabs(cpu - gpu) < 1.0e-3f ? EXIT_SUCCESS : EXIT_FAILURE;
}
''')

CODE[22] = simple_template(22, DATA[22][1], r'''
__global__ void squared_sum_partials_kernel(const float* x, float* partials, int n) {
  __shared__ float scratch[256];
  int global = blockIdx.x * blockDim.x + threadIdx.x;
  int local = threadIdx.x;
  float value = global < n ? x[global] * x[global] : 0.0f;
  scratch[local] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local < stride) scratch[local] += scratch[local + stride];
    __syncthreads();
  }
  if (local == 0) partials[blockIdx.x] = scratch[0];
}

int main() {
  const int n = 2048, threads = 256, blocks = (n + threads - 1) / threads;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  std::vector<float> x(n), partials(blocks, 0.0f);
  float cpu_sq = 0.0f;
  for (int i = 0; i < n; ++i) { x[i] = static_cast<float>((i % 21) - 10) * 0.125f; cpu_sq += x[i] * x[i]; }
  float *dx = nullptr, *dp = nullptr;
  CHECK_CUDA(cudaMalloc(&dx, bytes)); CHECK_CUDA(cudaMalloc(&dp, blocks * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice));
  squared_sum_partials_kernel<<<blocks, threads>>>(dx, dp, n);
  CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(partials.data(), dp, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  float gpu_sq = std::accumulate(partials.begin(), partials.end(), 0.0f);
  float cpu = std::sqrt(cpu_sq), gpu = std::sqrt(gpu_sq);
  std::cout << "CPU L2: " << cpu << "\nGPU L2: " << gpu << std::endl;
  std::cout << "Validation: " << (std::fabs(cpu - gpu) < 1.0e-3f ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(dx)); CHECK_CUDA(cudaFree(dp));
  return std::fabs(cpu - gpu) < 1.0e-3f ? EXIT_SUCCESS : EXIT_FAILURE;
}
''')

CODE[23] = simple_template(23, DATA[23][1], r'''
__global__ void sum_partials_kernel(const float* input, float* partials, int n) {
  __shared__ float scratch[256];
  int global = blockIdx.x * blockDim.x + threadIdx.x;
  int local = threadIdx.x;
  scratch[local] = global < n ? input[global] : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local < stride) scratch[local] += scratch[local + stride];
    __syncthreads();
  }
  if (local == 0) partials[blockIdx.x] = scratch[0];
}

int main() {
  const int n = 4096, threads = 256, blocks = (n + threads - 1) / threads;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  std::vector<float> input(n), partials(blocks, 0.0f);
  float cpu = 0.0f;
  for (int i = 0; i < n; ++i) { input[i] = static_cast<float>((i % 13) - 6); cpu += input[i]; }
  float *di = nullptr, *dp = nullptr;
  CHECK_CUDA(cudaMalloc(&di, bytes)); CHECK_CUDA(cudaMalloc(&dp, blocks * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(di, input.data(), bytes, cudaMemcpyHostToDevice));
  sum_partials_kernel<<<blocks, threads>>>(di, dp, n);
  CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(partials.data(), dp, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  float gpu = std::accumulate(partials.begin(), partials.end(), 0.0f);
  std::cout << "Validation: " << (std::fabs(cpu - gpu) < 1.0e-3f ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(dp));
  return std::fabs(cpu - gpu) < 1.0e-3f ? EXIT_SUCCESS : EXIT_FAILURE;
}
''')

CODE[24] = simple_template(24, DATA[24][1], r'''
__global__ void max_partials_kernel(const float* input, float* partials, int n) {
  __shared__ float scratch[256];
  int global = blockIdx.x * blockDim.x + threadIdx.x;
  int local = threadIdx.x;
  scratch[local] = global < n ? input[global] : -FLT_MAX;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local < stride) scratch[local] = fmaxf(scratch[local], scratch[local + stride]);
    __syncthreads();
  }
  if (local == 0) partials[blockIdx.x] = scratch[0];
}

int main() {
  const int n = 2048, threads = 256, blocks = (n + threads - 1) / threads;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  std::vector<float> input(n), partials(blocks, 0.0f);
  for (int i = 0; i < n; ++i) input[i] = static_cast<float>((i % 37) - 18);
  input[777] = 999.0f;
  float cpu = *std::max_element(input.begin(), input.end());
  float *di = nullptr, *dp = nullptr;
  CHECK_CUDA(cudaMalloc(&di, bytes)); CHECK_CUDA(cudaMalloc(&dp, blocks * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(di, input.data(), bytes, cudaMemcpyHostToDevice));
  max_partials_kernel<<<blocks, threads>>>(di, dp, n);
  CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(partials.data(), dp, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  float gpu = *std::max_element(partials.begin(), partials.end());
  std::cout << "Validation: " << (std::fabs(cpu - gpu) < 1.0e-5f ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(dp));
  return std::fabs(cpu - gpu) < 1.0e-5f ? EXIT_SUCCESS : EXIT_FAILURE;
}
''')

CODE[25] = simple_template(25, DATA[25][1], r'''
__global__ void min_partials_kernel(const float* input, float* partials, int n) {
  __shared__ float scratch[256];
  int global = blockIdx.x * blockDim.x + threadIdx.x;
  int local = threadIdx.x;
  scratch[local] = global < n ? input[global] : FLT_MAX;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local < stride) scratch[local] = fminf(scratch[local], scratch[local + stride]);
    __syncthreads();
  }
  if (local == 0) partials[blockIdx.x] = scratch[0];
}

int main() {
  const int n = 2048, threads = 256, blocks = (n + threads - 1) / threads;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  std::vector<float> input(n), partials(blocks, 0.0f);
  for (int i = 0; i < n; ++i) input[i] = static_cast<float>((i % 41) - 20);
  input[333] = -999.0f;
  float cpu = *std::min_element(input.begin(), input.end());
  float *di = nullptr, *dp = nullptr;
  CHECK_CUDA(cudaMalloc(&di, bytes)); CHECK_CUDA(cudaMalloc(&dp, blocks * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(di, input.data(), bytes, cudaMemcpyHostToDevice));
  min_partials_kernel<<<blocks, threads>>>(di, dp, n);
  CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(partials.data(), dp, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  float gpu = *std::min_element(partials.begin(), partials.end());
  std::cout << "Validation: " << (std::fabs(cpu - gpu) < 1.0e-5f ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(dp));
  return std::fabs(cpu - gpu) < 1.0e-5f ? EXIT_SUCCESS : EXIT_FAILURE;
}
''')
from textwrap import dedent

CODE[26] = simple_template(26, DATA[26][1], r'''
__global__ void hillis_steele_kernel(const int* input, int* output, int n) {
  __shared__ int data[256];
  int tid = threadIdx.x;
  data[tid] = tid < n ? input[tid] : 0;
  __syncthreads();
  for (int offset = 1; offset < n; offset <<= 1) {
    int add = tid >= offset ? data[tid - offset] : 0;
    __syncthreads();
    if (tid < n) data[tid] += add;
    __syncthreads();
  }
  if (tid < n) output[tid] = data[tid];
}
int main() { const int n = 128; std::vector<int> input(n), gpu(n,0), cpu(n,0); for(int i=0;i<n;++i){input[i]=(i%5)+1; cpu[i]=input[i]+(i?cpu[i-1]:0);} int *di=nullptr,*do_=nullptr; CHECK_CUDA(cudaMalloc(&di,n*sizeof(int))); CHECK_CUDA(cudaMalloc(&do_,n*sizeof(int))); CHECK_CUDA(cudaMemcpy(di,input.data(),n*sizeof(int),cudaMemcpyHostToDevice)); hillis_steele_kernel<<<1,256>>>(di,do_,n); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,n*sizeof(int),cudaMemcpyDeviceToHost)); bool ok=gpu==cpu; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(do_)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[27] = simple_template(27, DATA[27][1], r'''
__global__ void blelloch_inclusive_kernel(const int* input, int* output, int n) {
  __shared__ int temp[256];
  int tid = threadIdx.x;
  temp[tid] = tid < n ? input[tid] : 0;
  __syncthreads();
  for (int stride = 1; stride < n; stride <<= 1) {
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < n) temp[idx] += temp[idx - stride];
    __syncthreads();
  }
  if (tid == 0) temp[n - 1] = 0;
  __syncthreads();
  for (int stride = n >> 1; stride > 0; stride >>= 1) {
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < n) { int t = temp[idx - stride]; temp[idx - stride] = temp[idx]; temp[idx] += t; }
    __syncthreads();
  }
  if (tid < n) output[tid] = temp[tid] + input[tid];
}
int main() { const int n = 128; std::vector<int> input(n), gpu(n,0), cpu(n,0); for(int i=0;i<n;++i){input[i]=(i%7)+1; cpu[i]=input[i]+(i?cpu[i-1]:0);} int *di=nullptr,*do_=nullptr; CHECK_CUDA(cudaMalloc(&di,n*sizeof(int))); CHECK_CUDA(cudaMalloc(&do_,n*sizeof(int))); CHECK_CUDA(cudaMemcpy(di,input.data(),n*sizeof(int),cudaMemcpyHostToDevice)); blelloch_inclusive_kernel<<<1,256>>>(di,do_,n); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,n*sizeof(int),cudaMemcpyDeviceToHost)); bool ok=gpu==cpu; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(do_)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[28] = simple_template(28, DATA[28][1], r'''
__global__ void histogram_global_kernel(const unsigned int* input, unsigned int* bins, int n, int num_bins) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) atomicAdd(&bins[input[idx] % num_bins], 1u);
}
int main() { const int n=2048, num_bins=16; std::vector<unsigned int> input(n), gpu(num_bins,0), cpu(num_bins,0); for(int i=0;i<n;++i){input[i]=(i*7)%num_bins; ++cpu[input[i]];} unsigned int *di=nullptr,*db=nullptr; CHECK_CUDA(cudaMalloc(&di,n*sizeof(unsigned int))); CHECK_CUDA(cudaMalloc(&db,num_bins*sizeof(unsigned int))); CHECK_CUDA(cudaMemcpy(di,input.data(),n*sizeof(unsigned int),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemset(db,0,num_bins*sizeof(unsigned int))); histogram_global_kernel<<<(n+255)/256,256>>>(di,db,n,num_bins); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),db,num_bins*sizeof(unsigned int),cudaMemcpyDeviceToHost)); bool ok=gpu==cpu; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(db)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[29] = simple_template(29, DATA[29][1], r'''
__global__ void histogram_shared_kernel(const unsigned int* input, unsigned int* bins, int n, int num_bins) {
  __shared__ unsigned int local_bins[16];
  int tid = threadIdx.x;
  if (tid < num_bins) local_bins[tid] = 0;
  __syncthreads();
  int idx = blockIdx.x * blockDim.x + tid;
  if (idx < n) atomicAdd(&local_bins[input[idx] % num_bins], 1u);
  __syncthreads();
  if (tid < num_bins) atomicAdd(&bins[tid], local_bins[tid]);
}
int main() { const int n=2048, num_bins=16; std::vector<unsigned int> input(n), gpu(num_bins,0), cpu(num_bins,0); for(int i=0;i<n;++i){input[i]=((i*7)+(i/5))%num_bins; ++cpu[input[i]];} unsigned int *di=nullptr,*db=nullptr; CHECK_CUDA(cudaMalloc(&di,n*sizeof(unsigned int))); CHECK_CUDA(cudaMalloc(&db,num_bins*sizeof(unsigned int))); CHECK_CUDA(cudaMemcpy(di,input.data(),n*sizeof(unsigned int),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemset(db,0,num_bins*sizeof(unsigned int))); histogram_shared_kernel<<<(n+255)/256,256>>>(di,db,n,num_bins); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),db,num_bins*sizeof(unsigned int),cudaMemcpyDeviceToHost)); bool ok=gpu==cpu; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(db)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[30] = simple_template(30, DATA[30][1], r'''
__global__ void compact_positive_kernel(const int* input, int* output, int* count, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && input[idx] > 0) {
    int slot = atomicAdd(count, 1);
    output[slot] = input[idx];
  }
}
int main() { const int n=64; std::vector<int> input(n), cpu; for(int i=0;i<n;++i){input[i]=(i%9)-4; if(input[i]>0) cpu.push_back(input[i]);} std::vector<int> gpu(cpu.size(),0); int *di=nullptr,*do_=nullptr,*dc=nullptr; CHECK_CUDA(cudaMalloc(&di,n*sizeof(int))); CHECK_CUDA(cudaMalloc(&do_,n*sizeof(int))); CHECK_CUDA(cudaMalloc(&dc,sizeof(int))); CHECK_CUDA(cudaMemcpy(di,input.data(),n*sizeof(int),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemset(dc,0,sizeof(int))); compact_positive_kernel<<<1,128>>>(di,do_,dc,n); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); int count=0; CHECK_CUDA(cudaMemcpy(&count,dc,sizeof(int),cudaMemcpyDeviceToHost)); gpu.resize(count); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,count*sizeof(int),cudaMemcpyDeviceToHost)); bool ok=gpu==cpu; std::cout<<"Kept "<<count<<" elements\nValidation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(do_)); CHECK_CUDA(cudaFree(dc)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[31] = simple_template(31, DATA[31][1], r'''
__global__ void gather_kernel(const float* source, const int* indices, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) output[idx] = source[indices[idx]];
}
int main() { const int n=32; std::vector<float> source(64), gpu(n,0.0f), cpu(n,0.0f); std::vector<int> idxs(n); for(int i=0;i<64;++i) source[i]=i*1.5f; for(int i=0;i<n;++i){idxs[i]=(i*3)%64; cpu[i]=source[idxs[i]];} float *ds=nullptr,*do_=nullptr; int* di=nullptr; CHECK_CUDA(cudaMalloc(&ds,64*sizeof(float))); CHECK_CUDA(cudaMalloc(&di,n*sizeof(int))); CHECK_CUDA(cudaMalloc(&do_,n*sizeof(float))); CHECK_CUDA(cudaMemcpy(ds,source.data(),64*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(di,idxs.data(),n*sizeof(int),cudaMemcpyHostToDevice)); gather_kernel<<<1,128>>>(ds,di,do_,n); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,n*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(int i=0;i<n;++i) if(std::fabs(gpu[i]-cpu[i])>1e-5f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(ds)); CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(do_)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[32] = simple_template(32, DATA[32][1], r'''
__global__ void scatter_kernel(const float* input, const int* destinations, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) output[destinations[idx]] = input[idx];
}
int main() { const int n=32; std::vector<float> input(n), gpu(n,-1.0f), cpu(n,-1.0f); std::vector<int> dst(n); for(int i=0;i<n;++i){input[i]=100.0f+i; dst[i]=(i*5)%n; cpu[dst[i]]=input[i];} float *di=nullptr,*do_=nullptr; int* dd=nullptr; CHECK_CUDA(cudaMalloc(&di,n*sizeof(float))); CHECK_CUDA(cudaMalloc(&do_,n*sizeof(float))); CHECK_CUDA(cudaMalloc(&dd,n*sizeof(int))); CHECK_CUDA(cudaMemcpy(di,input.data(),n*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dd,dst.data(),n*sizeof(int),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemset(do_,0,n*sizeof(float))); scatter_kernel<<<1,128>>>(di,dd,do_,n); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,n*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(int i=0;i<n;++i) if(std::fabs(gpu[i]-cpu[i])>1e-5f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(do_)); CHECK_CUDA(cudaFree(dd)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[33] = simple_template(33, DATA[33][1], r'''
__global__ void count_positive_kernel(const int* input, int* count, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && input[idx] > 0) atomicAdd(count, 1);
}
int main() { const int n=1024; std::vector<int> input(n); int cpu=0; for(int i=0;i<n;++i){input[i]=(i%11)-5; if(input[i]>0) ++cpu;} int *di=nullptr,*dc=nullptr; int gpu=0; CHECK_CUDA(cudaMalloc(&di,n*sizeof(int))); CHECK_CUDA(cudaMalloc(&dc,sizeof(int))); CHECK_CUDA(cudaMemcpy(di,input.data(),n*sizeof(int),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemset(dc,0,sizeof(int))); count_positive_kernel<<<(n+255)/256,256>>>(di,dc,n); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(&gpu,dc,sizeof(int),cudaMemcpyDeviceToHost)); std::cout<<"Validation: "<<(gpu==cpu?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(dc)); return gpu==cpu?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[34] = simple_template(34, DATA[34][1], r'''
__global__ void find_first_kernel(const int* input, int target, int* first_index, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && input[idx] == target) atomicMin(first_index, idx);
}
int main() { const int n=512; const int target=42; std::vector<int> input(n); std::fill(input.begin(), input.end(), 7); input[137]=target; input[299]=target; int cpu=137, gpu=n; int *di=nullptr,*df=nullptr; CHECK_CUDA(cudaMalloc(&di,n*sizeof(int))); CHECK_CUDA(cudaMalloc(&df,sizeof(int))); CHECK_CUDA(cudaMemcpy(di,input.data(),n*sizeof(int),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(df,&gpu,sizeof(int),cudaMemcpyHostToDevice)); find_first_kernel<<<(n+255)/256,256>>>(di,target,df,n); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(&gpu,df,sizeof(int),cudaMemcpyDeviceToHost)); std::cout<<"Validation: "<<(gpu==cpu?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(df)); return gpu==cpu?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[35] = simple_template(35, DATA[35][1], r'''
__global__ void odd_even_phase_kernel(int* data, int n, int phase) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int i = 2 * tid + phase;
  if (i + 1 < n && data[i] > data[i + 1]) { int t = data[i]; data[i] = data[i + 1]; data[i + 1] = t; }
}
int main() { const int n=32; std::vector<int> input={9,4,1,7,3,8,2,6,5,0,11,10,13,12,15,14,19,16,18,17,21,20,23,22,25,24,27,26,29,28,31,30}; auto cpu=input; std::sort(cpu.begin(), cpu.end()); int* d=nullptr; CHECK_CUDA(cudaMalloc(&d,n*sizeof(int))); CHECK_CUDA(cudaMemcpy(d,input.data(),n*sizeof(int),cudaMemcpyHostToDevice)); for(int phase=0; phase<n; ++phase){ odd_even_phase_kernel<<<1,128>>>(d,n,phase&1); CHECK_CUDA(cudaGetLastError()); } CHECK_CUDA(cudaDeviceSynchronize()); std::vector<int> gpu(n); CHECK_CUDA(cudaMemcpy(gpu.data(),d,n*sizeof(int),cudaMemcpyDeviceToHost)); bool ok=gpu==cpu; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(d)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[36] = simple_template(36, DATA[36][1], r'''
__global__ void bitonic_step_kernel(int* data, int j, int k) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int ixj = i ^ j;
  if (ixj > i) {
    bool ascending = (i & k) == 0;
    if ((ascending && data[i] > data[ixj]) || (!ascending && data[i] < data[ixj])) {
      int t = data[i]; data[i] = data[ixj]; data[ixj] = t;
    }
  }
}
int main() { const int n=32; std::vector<int> input={23,1,17,9,3,15,7,13,31,29,27,25,21,19,11,5,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30}; auto cpu=input; std::sort(cpu.begin(), cpu.end()); int* d=nullptr; CHECK_CUDA(cudaMalloc(&d,n*sizeof(int))); CHECK_CUDA(cudaMemcpy(d,input.data(),n*sizeof(int),cudaMemcpyHostToDevice)); for(int k=2; k<=n; k<<=1) for(int j=k>>1; j>0; j>>=1){ bitonic_step_kernel<<<1,128>>>(d,j,k); CHECK_CUDA(cudaGetLastError()); } CHECK_CUDA(cudaDeviceSynchronize()); std::vector<int> gpu(n); CHECK_CUDA(cudaMemcpy(gpu.data(),d,n*sizeof(int),cudaMemcpyDeviceToHost)); bool ok=gpu==cpu; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(d)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[37] = simple_template(37, DATA[37][1], r'''
__global__ void compare_swap_pairs_kernel(int* data, const int* left, const int* right, int pair_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < pair_count) { int a = left[idx], b = right[idx]; if (data[a] > data[b]) { int t = data[a]; data[a] = data[b]; data[b] = t; } }
}
int main() { const int n=16; std::vector<int> input={15,3,14,2,13,1,12,0,11,7,10,6,9,5,8,4}; auto cpu=input; std::sort(cpu.begin(), cpu.end()); std::vector<std::pair<int,int>> pairs; for(int p=1;p<n;p*=2){ for(int i=0;i<n;i+=2*p){ for(int j=0;j<p && i+j+p<n;++j){ pairs.push_back({i+j,i+j+p}); } } }
  std::vector<int> left(pairs.size()), right(pairs.size()); for(size_t i=0;i<pairs.size();++i){ left[i]=pairs[i].first; right[i]=pairs[i].second; }
  int *d=nullptr,*dl=nullptr,*dr=nullptr; CHECK_CUDA(cudaMalloc(&d,n*sizeof(int))); CHECK_CUDA(cudaMalloc(&dl,left.size()*sizeof(int))); CHECK_CUDA(cudaMalloc(&dr,right.size()*sizeof(int))); CHECK_CUDA(cudaMemcpy(d,input.data(),n*sizeof(int),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dl,left.data(),left.size()*sizeof(int),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dr,right.data(),right.size()*sizeof(int),cudaMemcpyHostToDevice)); for(int pass=0; pass<n; ++pass){ compare_swap_pairs_kernel<<<1,128>>>(d,dl,dr,(int)left.size()); CHECK_CUDA(cudaGetLastError()); } CHECK_CUDA(cudaDeviceSynchronize()); std::vector<int> gpu(n); CHECK_CUDA(cudaMemcpy(gpu.data(),d,n*sizeof(int),cudaMemcpyDeviceToHost)); bool ok=gpu==cpu; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(d)); CHECK_CUDA(cudaFree(dl)); CHECK_CUDA(cudaFree(dr)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[38] = simple_template(38, DATA[38][1], r'''
__global__ void batched_binary_search_kernel(const int* data, const int* queries, int* positions, int chunk_size, int total_size, int query_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < query_count) {
    int q = queries[idx];
    int chunk = idx % (total_size / chunk_size);
    int lo = chunk * chunk_size, hi = lo + chunk_size - 1, pos = -1;
    while (lo <= hi) { int mid = (lo + hi) / 2; int v = data[mid]; if (v == q) { pos = mid; break; } if (v < q) lo = mid + 1; else hi = mid - 1; }
    positions[idx] = pos;
  }
}
int main() { const int chunk=8, chunks=4, total=chunk*chunks, qn=8; std::vector<int> data(total), queries(qn), gpu(qn,-1), cpu(qn,-1); for(int c=0;c<chunks;++c) for(int i=0;i<chunk;++i) data[c*chunk+i]=c*100+i*2; queries={0,6,100,108,200,214,300,314}; for(int i=0;i<qn;++i){int c=i%chunks; auto begin=data.begin()+c*chunk; auto end=begin+chunk; auto it=std::lower_bound(begin,end,queries[i]); cpu[i]=(it!=end && *it==queries[i])?(int)(it-data.begin()):-1;} int *dd=nullptr,*dq=nullptr,*dp=nullptr; CHECK_CUDA(cudaMalloc(&dd,total*sizeof(int))); CHECK_CUDA(cudaMalloc(&dq,qn*sizeof(int))); CHECK_CUDA(cudaMalloc(&dp,qn*sizeof(int))); CHECK_CUDA(cudaMemcpy(dd,data.data(),total*sizeof(int),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dq,queries.data(),qn*sizeof(int),cudaMemcpyHostToDevice)); batched_binary_search_kernel<<<1,128>>>(dd,dq,dp,chunk,total,qn); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),dp,qn*sizeof(int),cudaMemcpyDeviceToHost)); bool ok=gpu==cpu; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(dd)); CHECK_CUDA(cudaFree(dq)); CHECK_CUDA(cudaFree(dp)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[39] = simple_template(39, DATA[39][1], r'''
__global__ void merge_kernel(const int* a, int na, const int* b, int nb, int* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = na + nb;
  if (idx < total) {
    int lo = max(0, idx - nb);
    int hi = min(idx, na);
    while (lo < hi) {
      int mid = (lo + hi + 1) / 2;
      if (a[mid - 1] > b[idx - mid]) hi = mid - 1; else lo = mid;
    }
    int i = lo;
    int j = idx - i;
    int a_val = i < na ? a[i] : INT_MAX;
    int b_val = j < nb ? b[j] : INT_MAX;
    out[idx] = min(a_val, b_val);
  }
}
int main() { std::vector<int> a={1,4,7,10,13,16,19,22}; std::vector<int> b={0,2,3,5,6,8,9,11,12,14,15,17}; std::vector<int> cpu(a.size()+b.size()), gpu(cpu.size()); std::merge(a.begin(),a.end(),b.begin(),b.end(),cpu.begin()); int *da=nullptr,*db=nullptr,*do_=nullptr; CHECK_CUDA(cudaMalloc(&da,a.size()*sizeof(int))); CHECK_CUDA(cudaMalloc(&db,b.size()*sizeof(int))); CHECK_CUDA(cudaMalloc(&do_,cpu.size()*sizeof(int))); CHECK_CUDA(cudaMemcpy(da,a.data(),a.size()*sizeof(int),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(db,b.data(),b.size()*sizeof(int),cudaMemcpyHostToDevice)); merge_kernel<<<1,128>>>(da,(int)a.size(),db,(int)b.size(),do_); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,cpu.size()*sizeof(int),cudaMemcpyDeviceToHost)); bool ok=gpu==cpu; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(da)); CHECK_CUDA(cudaFree(db)); CHECK_CUDA(cudaFree(do_)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[40] = simple_template(40, DATA[40][1], r'''
__global__ void bitonic_step_kernel(int* data, int j, int k) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int ixj = i ^ j;
  if (ixj > i) {
    bool ascending = (i & k) == 0;
    if ((ascending && data[i] > data[ixj]) || (!ascending && data[i] < data[ixj])) {
      int t = data[i]; data[i] = data[ixj]; data[ixj] = t;
    }
  }
}
int main() { const int n=32, k_top=5; std::vector<int> input={12,99,3,47,18,76,5,65,23,88,14,54,67,31,42,90,1,72,8,60,27,81,36,95,11,58,69,20,84,7,52,40}; auto cpu=input; std::sort(cpu.begin(), cpu.end(), std::greater<int>()); std::vector<int> cpu_top(cpu.begin(), cpu.begin()+k_top); int* d=nullptr; CHECK_CUDA(cudaMalloc(&d,n*sizeof(int))); CHECK_CUDA(cudaMemcpy(d,input.data(),n*sizeof(int),cudaMemcpyHostToDevice)); for(int k=2; k<=n; k<<=1) for(int j=k>>1; j>0; j>>=1){ bitonic_step_kernel<<<1,128>>>(d,j,k); CHECK_CUDA(cudaGetLastError()); } CHECK_CUDA(cudaDeviceSynchronize()); std::vector<int> sorted(n); CHECK_CUDA(cudaMemcpy(sorted.data(),d,n*sizeof(int),cudaMemcpyDeviceToHost)); std::reverse(sorted.begin(), sorted.end()); std::vector<int> gpu_top(sorted.begin(), sorted.begin()+k_top); bool ok=gpu_top==cpu_top; std::cout<<"Top-k: "; for(int v:gpu_top) std::cout<<v<<' '; std::cout<<"\nValidation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(d)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

for i, (slug, title, _, _) in DATA.items():
    folder = EX / slug
    w(folder / 'README.md', readme(i))
    w(folder / 'main.cu', CODE[i])

