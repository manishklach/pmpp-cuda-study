// Example 070: Spectrogram With FFT
// Track: Image and Signal
// Difficulty: Advanced
// Status: Reference-friendly

#include <cuda_runtime.h>
#include <cmath>
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

constexpr float kPi = 3.14159265358979323846f;

__global__ void spectrogram_dft_kernel(const float *signal, float *spectrogram, int signal_length,
                                       int window_size, int hop_size, int freq_bins, int windows) {
  int window = blockIdx.x;
  int bin = threadIdx.x;
  if (window >= windows || bin >= freq_bins)
    return;
  int start = window * hop_size;
  float real = 0.0f, imag = 0.0f;
  for (int n = 0; n < window_size; ++n) {
    int index = start + n;
    if (index >= signal_length)
      break;
    float angle = -2.0f * kPi * bin * n / window_size;
    float sample = signal[index];
    real += sample * cosf(angle);
    imag += sample * sinf(angle);
  }
  spectrogram[window * freq_bins + bin] = real * real + imag * imag;
}

int main() {
  const int signal_length = 64, window_size = 16, hop_size = 8, freq_bins = 8;
  const int windows = 1 + (signal_length - window_size) / hop_size;
  std::vector<float> signal(signal_length), cpu(windows * freq_bins, 0.0f),
      gpu(windows * freq_bins, 0.0f);
  for (int i = 0; i < signal_length; ++i)
    signal[i] = sinf(2.0f * kPi * i / 8.0f) + 0.5f * sinf(2.0f * kPi * i / 16.0f);
  for (int w = 0; w < windows; ++w) {
    int start = w * hop_size;
    for (int bin = 0; bin < freq_bins; ++bin) {
      float real = 0.0f, imag = 0.0f;
      for (int n = 0; n < window_size; ++n) {
        float angle = -2.0f * kPi * bin * n / window_size;
        float sample = signal[start + n];
        real += sample * std::cos(angle);
        imag += sample * std::sin(angle);
      }
      cpu[w * freq_bins + bin] = real * real + imag * imag;
    }
  }

  float *d_signal = nullptr, *d_spectrogram = nullptr;
  CHECK_CUDA(cudaMalloc(&d_signal, signal.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_spectrogram, gpu.size() * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(d_signal, signal.data(), signal.size() * sizeof(float), cudaMemcpyHostToDevice));
  spectrogram_dft_kernel<<<windows, freq_bins>>>(d_signal, d_spectrogram, signal_length,
                                                 window_size, hop_size, freq_bins, windows);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(
      cudaMemcpy(gpu.data(), d_spectrogram, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (std::size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs(cpu[i] - gpu[i]) > 1.0e-3f)
      ok = false;
  std::cout << "Window count: " << windows << ", freq bins: " << freq_bins << std::endl;
  std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << std::endl;
  CHECK_CUDA(cudaFree(d_signal));
  CHECK_CUDA(cudaFree(d_spectrogram));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
