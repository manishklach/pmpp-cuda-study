from pathlib import Path
from textwrap import dedent

ROOT = Path(r"C:\Users\ManishKL\Documents\Playground\pmpp-cuda-study")
EX = ROOT / "examples"

DATA = {
41:("041_matrix-vector-multiply","Matrix Vector Multiply",["row-wise decomposition","dense GEMV baseline","CPU reference checks"],["Try rectangular matrices.","Use shared memory for the vector later."]),
42:("042_naive-matrix-multiply","Naive Matrix Multiply",["2D output mapping","global memory baseline","correctness first"],["Change matrix sizes.","Compare against the tiled version."]),
43:("043_tiled-matrix-multiply","Tiled Matrix Multiply",["shared-memory tiles","data reuse","block-level synchronization"],["Try tile sizes 8, 16, and 32.","Compare numerical output with the naive kernel."]),
44:("044_batched-matrix-multiply","Batched Matrix Multiply",["batch dimension indexing","independent GEMMs","launch shape design"],["Increase batch size.","Compare serial launches versus batched indexing."]),
45:("045_convolution-1d","Convolution 1D",["windowed access","kernel radius","boundary handling"],["Use a wider filter.","Move coefficients to constant memory later."]),
46:("046_convolution-2d","Convolution 2D",["2D neighborhoods","filter loops","image-style indexing"],["Swap in a sharpening kernel.","Try non-square filters."]),
47:("047_separable-convolution","Separable Convolution",["two-pass filtering","algorithmic speedup","intermediate buffers"],["Compare against direct 2D convolution.","Use different horizontal and vertical kernels."]),
48:("048_sobel-edge-detection","Sobel Edge Detection",["paired convolution passes","gradient magnitude","edge maps"],["Threshold the gradient map.","Compare absolute-sum versus Euclidean magnitude."]),
49:("049_gaussian-blur","Gaussian Blur",["weighted smoothing","normalized kernels","image denoising"],["Increase the radius.","Compare separable and direct blur variants."]),
50:("050_median-filter","Median Filter",["small-window sorting","nonlinear filtering","impulse-noise removal"],["Try a 5x5 window later.","Compare against Gaussian blur on noisy input."]),
51:("051_box-filter-with-shared-memory","Box Filter With Shared Memory",["tile loading","halo regions","shared-memory reuse"],["Change tile dimensions.","Compare with a global-memory box filter."]),
52:("052_sparse-matrix-vector-multiply-csr","Sparse Matrix Vector Multiply CSR",["CSR layout","one-row-per-thread mapping","irregular memory access"],["Change sparsity patterns.","Compare against a dense fallback."]),
53:("053_sparse-matrix-dense-vector-multiply","Sparse Matrix Dense Vector Multiply",["sparse-dense interaction","row traversal","format awareness"],["Try a different sparse layout later.","Increase rows and sparsity."]),
54:("054_jacobi-iteration","Jacobi Iteration",["iterative solvers","ping-pong buffers","convergence checks"],["Run more iterations.","Measure residual error after each iteration."]),
55:("055_red-black-relaxation","Red Black Relaxation",["checkerboard updates","dependency splitting","grid solvers"],["Use a larger grid.","Compare with Jacobi on the same stencil."]),
56:("056_power-iteration","Power Iteration",["repeated matvecs","vector normalization","dominant eigenvector estimation"],["Track Rayleigh quotient per iteration.","Try a matrix with a clearer dominant eigenvalue."]),
57:("057_lu-factorization-sketch","LU Factorization Sketch",["blocked factorization workflow","panel updates","library integration planning"],["Map the panel/update steps to kernels.","Compare hand-written pieces with cuSOLVER later."]),
58:("058_cholesky-factorization","Cholesky Factorization",["SPD assumptions","triangular updates","factorization validation"],["Build a tiny SPD matrix and validate on CPU.","Compare direct code with cuSOLVER later."]),
59:("059_qr-factorization-sketch","QR Factorization Sketch",["Householder reflections","orthogonalization workflow","library handoff points"],["Derive the CPU reference first.","Identify which parts belong in cuSOLVER."]),
60:("060_fft-based-convolution","FFT Based Convolution",["frequency-domain multiplication","padding strategy","FFT library workflow"],["Document the exact cuFFT calls needed.","Compare direct and FFT convolution crossover points."]),
}


def write(path: Path, text: str):
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")


def readme(i: int) -> str:
    slug, title, focus, modify = DATA[i]
    status = 'Reference-friendly' if i <= 56 else 'Guided template'
    lines=[f"# {i:03d} - {title}","",'- Track: `Linear Algebra`',f"- Difficulty: `{'Intermediate' if i<=50 else 'Advanced'}`",f"- Status: `{status}`",'- GitHub batch: `041-060`',"","## Goal","",f"Build and study a {'working CUDA implementation' if i<=56 else 'library-aware study scaffold'} of **{title}**.","","## PMPP Ideas To Focus On",""]
    lines.extend(f"- {x}" for x in focus)
    lines.extend(["","## Build","","```powershell","nvcc -std=c++17 -O2 main.cu -o example.exe","```","","## Run","","```powershell",".\\example.exe","```",""])
    if i<=56:
        lines.extend(["## Validation","","- The program prints `PASS` when GPU output matches the CPU reference or stays within tolerance.","- Start with the included tiny matrices before scaling up.",""])
    else:
        lines.extend(["## Study Notes","","- This example is intentionally a stronger study scaffold because it typically depends on CUDA math libraries not available in this authoring environment.","- Use the README plus code comments as a roadmap for the eventual implementation.",""])
    lines.extend(["## What To Modify Next",""])
    lines.extend(f"- {x}" for x in modify)
    return "\n".join(lines)

COMMON = """
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

def head(i,title):
    diff='Intermediate' if i<=50 else 'Advanced'
    return f"// Example {i:03d}: {title}\n// Difficulty: {diff}\n"+COMMON


def generic_library_stub(i,title,notes):
    return head(i,title).replace('Reference-friendly','Guided template') + dedent(f"""
    // This example is a study scaffold for a library-heavy linear algebra workflow.
    // Focus areas:
    {notes}

    int main() {{
      std::cout << "{i:03d} - {title}" << std::endl;
      std::cout << "This example is intentionally a scaffold for a CUDA-library-backed workflow." << std::endl;
      std::cout << "Validation: REVIEW STUDY NOTES" << std::endl;
      return EXIT_SUCCESS;
    }}
    """)
from textwrap import dedent

CODE = {}

CODE[41] = head(41, DATA[41][1]) + dedent(r'''
__global__ void matvec_kernel(const float* matrix, const float* vector, float* output, int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) sum += matrix[row * cols + col] * vector[col];
    output[row] = sum;
  }
}
int main() {
  const int rows = 16, cols = 8;
  std::vector<float> matrix(rows * cols), vector(cols), gpu(rows, 0.0f), cpu(rows, 0.0f);
  for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) matrix[r * cols + c] = static_cast<float>((r + c) % 7 + 1);
  for (int c = 0; c < cols; ++c) vector[c] = static_cast<float>(c + 1);
  for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) cpu[r] += matrix[r * cols + c] * vector[c];
  float *dm=nullptr,*dv=nullptr,*do_=nullptr; CHECK_CUDA(cudaMalloc(&dm,matrix.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&dv,vector.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&do_,gpu.size()*sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dm,matrix.data(),matrix.size()*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dv,vector.data(),vector.size()*sizeof(float),cudaMemcpyHostToDevice));
  matvec_kernel<<<1,64>>>(dm,dv,do_,rows,cols); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost));
  bool ok=true; for(int i=0;i<rows;++i) if(std::fabs(gpu[i]-cpu[i])>1e-5f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(dm)); CHECK_CUDA(cudaFree(dv)); CHECK_CUDA(cudaFree(do_)); return ok?EXIT_SUCCESS:EXIT_FAILURE;
}
''')

CODE[42] = head(42, DATA[42][1]) + dedent(r'''
__global__ void matmul_naive_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    float sum = 0.0f;
    for (int e = 0; e < k; ++e) sum += a[row * k + e] * b[e * n + col];
    c[row * n + col] = sum;
  }
}
int main(){ const int m=8,n=8,k=8; std::vector<float>a(m*k),b(k*n),gpu(m*n,0.0f),cpu(m*n,0.0f); for(int i=0;i<m*k;++i)a[i]=(i%5)+1; for(int i=0;i<k*n;++i)b[i]=(i%7)+1; for(int r=0;r<m;++r) for(int c=0;c<n;++c) for(int e=0;e<k;++e) cpu[r*n+c]+=a[r*k+e]*b[e*n+c]; float *da=nullptr,*db=nullptr,*dc=nullptr; CHECK_CUDA(cudaMalloc(&da,a.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&db,b.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&dc,gpu.size()*sizeof(float))); CHECK_CUDA(cudaMemcpy(da,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(db,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice)); dim3 t(16,16), bl((n+t.x-1)/t.x,(m+t.y-1)/t.y); matmul_naive_kernel<<<bl,t>>>(da,db,dc,m,n,k); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),dc,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(int i=0;i<m*n;++i) if(std::fabs(gpu[i]-cpu[i])>1e-5f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(da)); CHECK_CUDA(cudaFree(db)); CHECK_CUDA(cudaFree(dc)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[43] = head(43, DATA[43][1]) + dedent(r'''
constexpr int TILE = 16;
__global__ void matmul_tiled_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
  __shared__ float tile_a[TILE][TILE];
  __shared__ float tile_b[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.0f;
  for (int t = 0; t < (k + TILE - 1) / TILE; ++t) {
    int a_col = t * TILE + threadIdx.x;
    int b_row = t * TILE + threadIdx.y;
    tile_a[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();
    for (int e = 0; e < TILE; ++e) sum += tile_a[threadIdx.y][e] * tile_b[e][threadIdx.x];
    __syncthreads();
  }
  if (row < m && col < n) c[row * n + col] = sum;
}
int main(){ const int m=16,n=16,k=16; std::vector<float>a(m*k),b(k*n),gpu(m*n,0.0f),cpu(m*n,0.0f); for(int i=0;i<m*k;++i)a[i]=(i%5)+1; for(int i=0;i<k*n;++i)b[i]=(i%7)+1; for(int r=0;r<m;++r) for(int c=0;c<n;++c) for(int e=0;e<k;++e) cpu[r*n+c]+=a[r*k+e]*b[e*n+c]; float *da=nullptr,*db=nullptr,*dc=nullptr; CHECK_CUDA(cudaMalloc(&da,a.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&db,b.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&dc,gpu.size()*sizeof(float))); CHECK_CUDA(cudaMemcpy(da,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(db,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice)); dim3 t(TILE,TILE), bl((n+TILE-1)/TILE,(m+TILE-1)/TILE); matmul_tiled_kernel<<<bl,t>>>(da,db,dc,m,n,k); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),dc,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(int i=0;i<m*n;++i) if(std::fabs(gpu[i]-cpu[i])>1e-5f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(da)); CHECK_CUDA(cudaFree(db)); CHECK_CUDA(cudaFree(dc)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[44] = head(44, DATA[44][1]) + dedent(r'''
__global__ void batched_matmul_kernel(const float* a, const float* b, float* c, int batch, int m, int n, int k) {
  int batch_id = blockIdx.z;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (batch_id < batch && row < m && col < n) {
    const float* a_batch = a + batch_id * m * k;
    const float* b_batch = b + batch_id * k * n;
    float* c_batch = c + batch_id * m * n;
    float sum = 0.0f;
    for (int e = 0; e < k; ++e) sum += a_batch[row * k + e] * b_batch[e * n + col];
    c_batch[row * n + col] = sum;
  }
}
int main(){ const int batch=3,m=4,n=4,k=4; std::vector<float>a(batch*m*k),b(batch*k*n),gpu(batch*m*n,0.0f),cpu(batch*m*n,0.0f); for(size_t i=0;i<a.size();++i)a[i]=(i%5)+1; for(size_t i=0;i<b.size();++i)b[i]=(i%7)+1; for(int bt=0;bt<batch;++bt) for(int r=0;r<m;++r) for(int c=0;c<n;++c) for(int e=0;e<k;++e) cpu[bt*m*n+r*n+c]+=a[bt*m*k+r*k+e]*b[bt*k*n+e*n+c]; float *da=nullptr,*db=nullptr,*dc=nullptr; CHECK_CUDA(cudaMalloc(&da,a.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&db,b.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&dc,gpu.size()*sizeof(float))); CHECK_CUDA(cudaMemcpy(da,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(db,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice)); dim3 t(16,16), bl((n+t.x-1)/t.x,(m+t.y-1)/t.y,batch); batched_matmul_kernel<<<bl,t>>>(da,db,dc,batch,m,n,k); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),dc,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(size_t i=0;i<gpu.size();++i) if(std::fabs(gpu[i]-cpu[i])>1e-5f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(da)); CHECK_CUDA(cudaFree(db)); CHECK_CUDA(cudaFree(dc)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[45] = head(45, DATA[45][1]) + dedent(r'''
__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, int n, int radius) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
      int x = min(max(idx + k, 0), n - 1);
      sum += input[x] * kernel[k + radius];
    }
    output[idx] = sum;
  }
}
int main(){ const int n=64,r=1; std::vector<float>in(n),ker={0.25f,0.5f,0.25f},gpu(n,0.0f),cpu(n,0.0f); for(int i=0;i<n;++i) in[i]=(float)(i%9); for(int i=0;i<n;++i) for(int k=-r;k<=r;++k){int x=std::min(std::max(i+k,0),n-1); cpu[i]+=in[x]*ker[k+r];} float *di=nullptr,*dk=nullptr,*do_=nullptr; CHECK_CUDA(cudaMalloc(&di,n*sizeof(float))); CHECK_CUDA(cudaMalloc(&dk,ker.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&do_,n*sizeof(float))); CHECK_CUDA(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dk,ker.data(),ker.size()*sizeof(float),cudaMemcpyHostToDevice)); conv1d_kernel<<<1,128>>>(di,dk,do_,n,r); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,n*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(int i=0;i<n;++i) if(std::fabs(gpu[i]-cpu[i])>1e-5f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(dk)); CHECK_CUDA(cudaFree(do_)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[46] = head(46, DATA[46][1]) + dedent(r'''
__global__ void conv2d_kernel(const float* input, const float* kernel, float* output, int w, int h, int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky) {
      for (int kx = -radius; kx <= radius; ++kx) {
        int sx = min(max(x + kx, 0), w - 1);
        int sy = min(max(y + ky, 0), h - 1);
        sum += input[sy * w + sx] * kernel[(ky + radius) * (2 * radius + 1) + (kx + radius)];
      }
    }
    output[y * w + x] = sum;
  }
}
int main(){ const int w=8,h=8,r=1; std::vector<float>in(w*h),ker={0,1,0,1,4,1,0,1,0},gpu(w*h,0.0f),cpu(w*h,0.0f); for(int i=0;i<w*h;++i) in[i]=(float)((i%7)+1); for(float &k:ker) k/=8.0f; for(int y=0;y<h;++y) for(int x=0;x<w;++x) for(int ky=-r;ky<=r;++ky) for(int kx=-r;kx<=r;++kx){int sx=std::min(std::max(x+kx,0),w-1); int sy=std::min(std::max(y+ky,0),h-1); cpu[y*w+x]+=in[sy*w+sx]*ker[(ky+r)*3+(kx+r)];} float *di=nullptr,*dk=nullptr,*do_=nullptr; CHECK_CUDA(cudaMalloc(&di,in.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&dk,ker.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&do_,gpu.size()*sizeof(float))); CHECK_CUDA(cudaMemcpy(di,in.data(),in.size()*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dk,ker.data(),ker.size()*sizeof(float),cudaMemcpyHostToDevice)); dim3 t(16,16), bl((w+t.x-1)/t.x,(h+t.y-1)/t.y); conv2d_kernel<<<bl,t>>>(di,dk,do_,w,h,r); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(size_t i=0;i<gpu.size();++i) if(std::fabs(gpu[i]-cpu[i])>1e-5f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(dk)); CHECK_CUDA(cudaFree(do_)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')
from textwrap import dedent

CODE[47] = head(47, DATA[47][1]) + dedent(r'''
__global__ void conv_horizontal_kernel(const float* input, const float* kernel, float* output, int w, int h, int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) { float sum = 0.0f; for (int k = -radius; k <= radius; ++k) { int sx = min(max(x + k, 0), w - 1); sum += input[y * w + sx] * kernel[k + radius]; } output[y * w + x] = sum; }
}
__global__ void conv_vertical_kernel(const float* input, const float* kernel, float* output, int w, int h, int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) { float sum = 0.0f; for (int k = -radius; k <= radius; ++k) { int sy = min(max(y + k, 0), h - 1); sum += input[sy * w + x] * kernel[k + radius]; } output[y * w + x] = sum; }
}
int main(){ const int w=8,h=8,r=1; std::vector<float>in(w*h), ker={0.25f,0.5f,0.25f}, temp(w*h,0.0f), gpu(w*h,0.0f), cpu(w*h,0.0f), cpu_temp(w*h,0.0f); for(int i=0;i<w*h;++i) in[i]=(float)((i%9)+1); for(int y=0;y<h;++y) for(int x=0;x<w;++x) for(int k=-r;k<=r;++k){int sx=std::min(std::max(x+k,0),w-1); cpu_temp[y*w+x]+=in[y*w+sx]*ker[k+r];} for(int y=0;y<h;++y) for(int x=0;x<w;++x) for(int k=-r;k<=r;++k){int sy=std::min(std::max(y+k,0),h-1); cpu[y*w+x]+=cpu_temp[sy*w+x]*ker[k+r];} float *di=nullptr,*dk=nullptr,*dt=nullptr,*do_=nullptr; CHECK_CUDA(cudaMalloc(&di,in.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&dk,ker.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&dt,temp.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&do_,gpu.size()*sizeof(float))); CHECK_CUDA(cudaMemcpy(di,in.data(),in.size()*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dk,ker.data(),ker.size()*sizeof(float),cudaMemcpyHostToDevice)); dim3 t(16,16), bl((w+t.x-1)/t.x,(h+t.y-1)/t.y); conv_horizontal_kernel<<<bl,t>>>(di,dk,dt,w,h,r); conv_vertical_kernel<<<bl,t>>>(dt,dk,do_,w,h,r); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(size_t i=0;i<gpu.size();++i) if(std::fabs(gpu[i]-cpu[i])>1e-5f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(dk)); CHECK_CUDA(cudaFree(dt)); CHECK_CUDA(cudaFree(do_)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[48] = head(48, DATA[48][1]) + dedent(r'''
__global__ void sobel_kernel(const float* input, float* output, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x > 0 && x + 1 < w && y > 0 && y + 1 < h) {
    int idx = y * w + x;
    float gx = -input[(y-1)*w + (x-1)] + input[(y-1)*w + (x+1)] - 2*input[y*w + (x-1)] + 2*input[y*w + (x+1)] - input[(y+1)*w + (x-1)] + input[(y+1)*w + (x+1)];
    float gy = -input[(y-1)*w + (x-1)] - 2*input[(y-1)*w + x] - input[(y-1)*w + (x+1)] + input[(y+1)*w + (x-1)] + 2*input[(y+1)*w + x] + input[(y+1)*w + (x+1)];
    output[idx] = sqrtf(gx*gx + gy*gy);
  }
}
int main(){ const int w=8,h=8; std::vector<float>in(w*h),gpu(w*h,0.0f),cpu(w*h,0.0f); for(int y=0;y<h;++y) for(int x=0;x<w;++x) in[y*w+x]=(x<4?0.0f:10.0f); for(int y=1;y<h-1;++y) for(int x=1;x<w-1;++x){float gx=-in[(y-1)*w+(x-1)] + in[(y-1)*w+(x+1)] -2*in[y*w+(x-1)] +2*in[y*w+(x+1)] -in[(y+1)*w+(x-1)] + in[(y+1)*w+(x+1)]; float gy=-in[(y-1)*w+(x-1)] -2*in[(y-1)*w+x] -in[(y-1)*w+(x+1)] + in[(y+1)*w+(x-1)] +2*in[(y+1)*w+x] + in[(y+1)*w+(x+1)]; cpu[y*w+x]=std::sqrt(gx*gx+gy*gy);} float *di=nullptr,*do_=nullptr; CHECK_CUDA(cudaMalloc(&di,in.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&do_,gpu.size()*sizeof(float))); CHECK_CUDA(cudaMemcpy(di,in.data(),in.size()*sizeof(float),cudaMemcpyHostToDevice)); dim3 t(16,16), bl((w+t.x-1)/t.x,(h+t.y-1)/t.y); sobel_kernel<<<bl,t>>>(di,do_,w,h); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(size_t i=0;i<gpu.size();++i) if(std::fabs(gpu[i]-cpu[i])>1e-4f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(do_)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[49] = head(49, DATA[49][1]) + dedent(r'''
__global__ void gaussian_kernel(const float* input, float* output, int w, int h) {
  __shared__ float k[9];
  if (threadIdx.x < 9 && threadIdx.y == 0) { float vals[9]={1,2,1,2,4,2,1,2,1}; k[threadIdx.x]=vals[threadIdx.x]/16.0f; }
  __syncthreads();
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x<w && y<h){ float sum=0.0f; for(int ky=-1;ky<=1;++ky) for(int kx=-1;kx<=1;++kx){int sx=min(max(x+kx,0),w-1); int sy=min(max(y+ky,0),h-1); sum+=input[sy*w+sx]*k[(ky+1)*3+(kx+1)];} output[y*w+x]=sum; }
}
int main(){ const int w=8,h=8; std::vector<float>in(w*h),gpu(w*h,0.0f),cpu(w*h,0.0f),k={1/16.0f,2/16.0f,1/16.0f,2/16.0f,4/16.0f,2/16.0f,1/16.0f,2/16.0f,1/16.0f}; for(int i=0;i<w*h;++i) in[i]=(float)((i%5)+1); for(int y=0;y<h;++y) for(int x=0;x<w;++x) for(int ky=-1;ky<=1;++ky) for(int kx=-1;kx<=1;++kx){int sx=std::min(std::max(x+kx,0),w-1); int sy=std::min(std::max(y+ky,0),h-1); cpu[y*w+x]+=in[sy*w+sx]*k[(ky+1)*3+(kx+1)];} float *di=nullptr,*do_=nullptr; CHECK_CUDA(cudaMalloc(&di,in.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&do_,gpu.size()*sizeof(float))); CHECK_CUDA(cudaMemcpy(di,in.data(),in.size()*sizeof(float),cudaMemcpyHostToDevice)); dim3 t(16,16), bl((w+t.x-1)/t.x,(h+t.y-1)/t.y); gaussian_kernel<<<bl,t>>>(di,do_,w,h); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(size_t i=0;i<gpu.size();++i) if(std::fabs(gpu[i]-cpu[i])>1e-5f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(do_)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[50] = head(50, DATA[50][1]) + dedent(r'''
__device__ void sort9(float* vals){ for(int i=0;i<9;++i) for(int j=i+1;j<9;++j) if(vals[j]<vals[i]){ float t=vals[i]; vals[i]=vals[j]; vals[j]=t; }}
__global__ void median3x3_kernel(const float* input, float* output, int w, int h) {
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x<w && y<h){ float vals[9]; int p=0; for(int ky=-1;ky<=1;++ky) for(int kx=-1;kx<=1;++kx){int sx=min(max(x+kx,0),w-1); int sy=min(max(y+ky,0),h-1); vals[p++]=input[sy*w+sx];} sort9(vals); output[y*w+x]=vals[4]; }
}
int main(){ const int w=8,h=8; std::vector<float>in(w*h),gpu(w*h,0.0f),cpu(w*h,0.0f); for(int i=0;i<w*h;++i) in[i]=(float)((i*3)%11); in[10]=99.0f; for(int y=0;y<h;++y) for(int x=0;x<w;++x){ float vals[9]; int p=0; for(int ky=-1;ky<=1;++ky) for(int kx=-1;kx<=1;++kx){int sx=std::min(std::max(x+kx,0),w-1); int sy=std::min(std::max(y+ky,0),h-1); vals[p++]=in[sy*w+sx];} std::sort(vals, vals+9); cpu[y*w+x]=vals[4]; } float *di=nullptr,*do_=nullptr; CHECK_CUDA(cudaMalloc(&di,in.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&do_,gpu.size()*sizeof(float))); CHECK_CUDA(cudaMemcpy(di,in.data(),in.size()*sizeof(float),cudaMemcpyHostToDevice)); dim3 t(16,16), bl((w+t.x-1)/t.x,(h+t.y-1)/t.y); median3x3_kernel<<<bl,t>>>(di,do_,w,h); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(size_t i=0;i<gpu.size();++i) if(std::fabs(gpu[i]-cpu[i])>1e-5f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(do_)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[51] = head(51, DATA[51][1]) + dedent(r'''
constexpr int TILE=8;
__global__ void box_filter_shared_kernel(const float* input, float* output, int w, int h) {
  __shared__ float tile[TILE+2][TILE+2];
  int tx=threadIdx.x, ty=threadIdx.y;
  int x=blockIdx.x*TILE+tx, y=blockIdx.y*TILE+ty;
  int sx=min(max(x-1,0),w-1), sy=min(max(y-1,0),h-1);
  tile[ty][tx]=input[sy*w+sx];
  if(tx<TILE && ty<TILE && x<w && y<h){ int cx=min(max(x,0),w-1), cy=min(max(y,0),h-1); tile[ty+1][tx+1]=input[cy*w+cx]; }
  __syncthreads();
  if(tx<TILE && ty<TILE && x<w && y<h){ float sum=0.0f; for(int ky=0;ky<3;++ky) for(int kx=0;kx<3;++kx) sum+=tile[ty+ky][tx+kx]; output[y*w+x]=sum/9.0f; }
}
int main(){ const int w=8,h=8; std::vector<float>in(w*h),gpu(w*h,0.0f),cpu(w*h,0.0f); for(int i=0;i<w*h;++i) in[i]=(float)((i%7)+1); for(int y=0;y<h;++y) for(int x=0;x<w;++x){ float sum=0.0f; for(int ky=-1;ky<=1;++ky) for(int kx=-1;kx<=1;++kx){int sx=std::min(std::max(x+kx,0),w-1); int sy=std::min(std::max(y+ky,0),h-1); sum+=in[sy*w+sx];} cpu[y*w+x]=sum/9.0f;} float *di=nullptr,*do_=nullptr; CHECK_CUDA(cudaMalloc(&di,in.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&do_,gpu.size()*sizeof(float))); CHECK_CUDA(cudaMemcpy(di,in.data(),in.size()*sizeof(float),cudaMemcpyHostToDevice)); dim3 t(TILE,TILE), bl((w+TILE-1)/TILE,(h+TILE-1)/TILE); box_filter_shared_kernel<<<bl,t>>>(di,do_,w,h); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(size_t i=0;i<gpu.size();++i) if(std::fabs(gpu[i]-cpu[i])>1e-4f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(di)); CHECK_CUDA(cudaFree(do_)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

for i in range(41,52):
    slug,title,_,_=DATA[i]
    folder=EX/slug
    write(folder/'README.md', readme(i))
    write(folder/'main.cu', CODE[i])

for i in range(57,61):
    slug,title,focus,_=DATA[i]
    notes='\n'.join(f'// - {item}' for item in focus)
    folder=EX/slug
    write(folder/'README.md', readme(i))
    write(folder/'main.cu', generic_library_stub(i,title,notes))
from textwrap import dedent

CODE[52] = head(52, DATA[52][1]) + dedent(r'''
__global__ void spmv_csr_kernel(const int* row_ptr, const int* col_idx, const float* values, const float* x, float* y, int rows) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int jj = row_ptr[row]; jj < row_ptr[row + 1]; ++jj) sum += values[jj] * x[col_idx[jj]];
    y[row] = sum;
  }
}
int main(){ const int rows=4, cols=4; std::vector<int> row_ptr={0,2,4,7,8}, col_idx={0,2,1,3,0,2,3,1}; std::vector<float> vals={10,2,3,9,7,8,7,5}, x={1,2,3,4}, gpu(rows,0.0f), cpu(rows,0.0f); for(int r=0;r<rows;++r) for(int jj=row_ptr[r]; jj<row_ptr[r+1]; ++jj) cpu[r]+=vals[jj]*x[col_idx[jj]]; int *dr=nullptr,*dc=nullptr; float *dv=nullptr,*dx=nullptr,*dy=nullptr; CHECK_CUDA(cudaMalloc(&dr,row_ptr.size()*sizeof(int))); CHECK_CUDA(cudaMalloc(&dc,col_idx.size()*sizeof(int))); CHECK_CUDA(cudaMalloc(&dv,vals.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&dx,x.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&dy,gpu.size()*sizeof(float))); CHECK_CUDA(cudaMemcpy(dr,row_ptr.data(),row_ptr.size()*sizeof(int),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dc,col_idx.data(),col_idx.size()*sizeof(int),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dv,vals.data(),vals.size()*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dx,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice)); spmv_csr_kernel<<<1,64>>>(dr,dc,dv,dx,dy,rows); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),dy,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(int i=0;i<rows;++i) if(std::fabs(gpu[i]-cpu[i])>1e-5f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(dr)); CHECK_CUDA(cudaFree(dc)); CHECK_CUDA(cudaFree(dv)); CHECK_CUDA(cudaFree(dx)); CHECK_CUDA(cudaFree(dy)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[53] = head(53, DATA[53][1]) + dedent(r'''
__global__ void sparse_dense_kernel(const int* row_ptr, const int* col_idx, const float* values, const float* dense, float* out, int rows, int dense_cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    for (int dc = 0; dc < dense_cols; ++dc) {
      float sum = 0.0f;
      for (int jj = row_ptr[row]; jj < row_ptr[row + 1]; ++jj) sum += values[jj] * dense[col_idx[jj] * dense_cols + dc];
      out[row * dense_cols + dc] = sum;
    }
  }
}
int main(){ const int rows=4,dense_cols=3; std::vector<int> row_ptr={0,2,4,7,8}, col_idx={0,2,1,3,0,2,3,1}; std::vector<float> vals={10,2,3,9,7,8,7,5}, dense={1,2,3,4,5,6,7,8,9,10,11,12}, gpu(rows*dense_cols,0.0f), cpu(rows*dense_cols,0.0f); for(int r=0;r<rows;++r) for(int dc=0;dc<dense_cols;++dc) for(int jj=row_ptr[r]; jj<row_ptr[r+1]; ++jj) cpu[r*dense_cols+dc]+=vals[jj]*dense[col_idx[jj]*dense_cols+dc]; int *dr=nullptr,*dcidx=nullptr; float *dv=nullptr,*dd=nullptr,*do_=nullptr; CHECK_CUDA(cudaMalloc(&dr,row_ptr.size()*sizeof(int))); CHECK_CUDA(cudaMalloc(&dcidx,col_idx.size()*sizeof(int))); CHECK_CUDA(cudaMalloc(&dv,vals.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&dd,dense.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&do_,gpu.size()*sizeof(float))); CHECK_CUDA(cudaMemcpy(dr,row_ptr.data(),row_ptr.size()*sizeof(int),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dcidx,col_idx.data(),col_idx.size()*sizeof(int),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dv,vals.data(),vals.size()*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dd,dense.data(),dense.size()*sizeof(float),cudaMemcpyHostToDevice)); sparse_dense_kernel<<<1,64>>>(dr,dcidx,dv,dd,do_,rows,dense_cols); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),do_,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(size_t i=0;i<gpu.size();++i) if(std::fabs(gpu[i]-cpu[i])>1e-5f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(dr)); CHECK_CUDA(cudaFree(dcidx)); CHECK_CUDA(cudaFree(dv)); CHECK_CUDA(cudaFree(dd)); CHECK_CUDA(cudaFree(do_)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[54] = head(54, DATA[54][1]) + dedent(r'''
__global__ void jacobi_step_kernel(const float* a, const float* b, const float* x_old, float* x_new, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) {
    float sigma = 0.0f;
    for (int col = 0; col < n; ++col) if (col != row) sigma += a[row * n + col] * x_old[col];
    x_new[row] = (b[row] - sigma) / a[row * n + row];
  }
}
int main(){ const int n=4, iters=10; std::vector<float>a={10,-1,2,0,-1,11,-1,3,2,-1,10,-1,0,3,-1,8}, b={6,25,-11,15}, x(n,0.0f), next(n,0.0f), cpu(n,0.0f), cpu_next(n,0.0f), gpu(n,0.0f); for(int it=0;it<iters;++it){ for(int r=0;r<n;++r){ float sigma=0.0f; for(int c=0;c<n;++c) if(c!=r) sigma+=a[r*n+c]*cpu[c]; cpu_next[r]=(b[r]-sigma)/a[r*n+r]; } cpu=cpu_next; } float *da=nullptr,*db=nullptr,*dx=nullptr,*dn=nullptr; CHECK_CUDA(cudaMalloc(&da,a.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&db,b.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&dx,x.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&dn,next.size()*sizeof(float))); CHECK_CUDA(cudaMemcpy(da,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(db,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dx,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice)); for(int it=0;it<iters;++it){ jacobi_step_kernel<<<1,64>>>(da,db,dx,dn,n); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); std::swap(dx,dn);} CHECK_CUDA(cudaMemcpy(gpu.data(),dx,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(int i=0;i<n;++i) if(std::fabs(gpu[i]-cpu[i])>1e-3f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(da)); CHECK_CUDA(cudaFree(db)); CHECK_CUDA(cudaFree(dx)); CHECK_CUDA(cudaFree(dn)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[55] = head(55, DATA[55][1]) + dedent(r'''
__global__ void red_black_step_kernel(const float* input, float* output, int w, int h, int color) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x > 0 && x + 1 < w && y > 0 && y + 1 < h && ((x + y) & 1) == color) {
    output[y * w + x] = 0.25f * (input[y * w + x - 1] + input[y * w + x + 1] + input[(y - 1) * w + x] + input[(y + 1) * w + x]);
  }
}
int main(){ const int w=8,h=8; std::vector<float>grid(w*h,0.0f), cpu(w*h,0.0f), tmp(w*h,0.0f), gpu(w*h,0.0f); for(int x=0;x<w;++x){grid[x]=1.0f; grid[(h-1)*w+x]=1.0f; cpu[x]=1.0f; cpu[(h-1)*w+x]=1.0f;} tmp=cpu; for(int color=0;color<2;++color) for(int y=1;y<h-1;++y) for(int x=1;x<w-1;++x) if(((x+y)&1)==color) tmp[y*w+x]=0.25f*(cpu[y*w+x-1]+cpu[y*w+x+1]+cpu[(y-1)*w+x]+cpu[(y+1)*w+x]); cpu=tmp; float *d0=nullptr,*d1=nullptr; CHECK_CUDA(cudaMalloc(&d0,grid.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&d1,grid.size()*sizeof(float))); CHECK_CUDA(cudaMemcpy(d0,grid.data(),grid.size()*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(d1,grid.data(),grid.size()*sizeof(float),cudaMemcpyHostToDevice)); dim3 t(16,16), bl((w+t.x-1)/t.x,(h+t.y-1)/t.y); red_black_step_kernel<<<bl,t>>>(d0,d1,w,h,0); red_black_step_kernel<<<bl,t>>>(d1,d1,w,h,1); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),d1,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost)); bool ok=true; for(size_t i=0;i<gpu.size();++i) if(std::fabs(gpu[i]-cpu[i])>1e-4f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(d0)); CHECK_CUDA(cudaFree(d1)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

CODE[56] = head(56, DATA[56][1]) + dedent(r'''
__global__ void matvec_power_kernel(const float* matrix, const float* x, float* y, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) {
    float sum = 0.0f;
    for (int col = 0; col < n; ++col) sum += matrix[row * n + col] * x[col];
    y[row] = sum;
  }
}
int main(){ const int n=3, iters=8; std::vector<float>A={4,1,1,1,3,0,1,0,2}, x(n,1.0f), y(n,0.0f), cpu_x(n,1.0f), cpu_y(n,0.0f), gpu(n,0.0f); for(int it=0;it<iters;++it){ for(int r=0;r<n;++r){ cpu_y[r]=0.0f; for(int c=0;c<n;++c) cpu_y[r]+=A[r*n+c]*cpu_x[c]; } float norm=0.0f; for(float v:cpu_y) norm+=v*v; norm=std::sqrt(norm); for(int i=0;i<n;++i) cpu_x[i]=cpu_y[i]/norm; } float *dA=nullptr,*dx=nullptr,*dy=nullptr; CHECK_CUDA(cudaMalloc(&dA,A.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&dx,x.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&dy,y.size()*sizeof(float))); CHECK_CUDA(cudaMemcpy(dA,A.data(),A.size()*sizeof(float),cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dx,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice)); for(int it=0;it<iters;++it){ matvec_power_kernel<<<1,64>>>(dA,dx,dy,n); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaMemcpy(gpu.data(),dy,gpu.size()*sizeof(float),cudaMemcpyDeviceToHost)); float norm=0.0f; for(float v:gpu) norm+=v*v; norm=std::sqrt(norm); for(float &v:gpu) v/=norm; CHECK_CUDA(cudaMemcpy(dx,gpu.data(),gpu.size()*sizeof(float),cudaMemcpyHostToDevice)); } bool ok=true; for(int i=0;i<n;++i) if(std::fabs(gpu[i]-cpu_x[i])>1e-3f) ok=false; std::cout<<"Validation: "<<(ok?"PASS":"FAIL")<<std::endl; CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dx)); CHECK_CUDA(cudaFree(dy)); return ok?EXIT_SUCCESS:EXIT_FAILURE; }
''')

for i in range(52,57):
    slug,title,_,_=DATA[i]
    folder=EX/slug
    write(folder/'README.md', readme(i))
    write(folder/'main.cu', CODE[i])
