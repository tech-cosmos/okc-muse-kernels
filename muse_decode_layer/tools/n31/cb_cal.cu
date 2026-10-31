// n31 harness calibration: cuBLAS GemmEx (BF16, FP32 accumulate) for out[M,2304] = x[M,8192] @ W^T, timed like cl_gemm.cu.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
int main(int argc, char** argv) {
  int M = atoi(argv[1]), N = 2304, K = 8192; __nv_bfloat16 *x, *w, *o; char* fl;
  cudaMalloc(&x, (size_t)M * K * 2); cudaMalloc(&w, (size_t)N * K * 2); cudaMalloc(&o, (size_t)M * N * 2); cudaMalloc(&fl, 200 << 20);
  cublasHandle_t h; cublasCreate(&h); float al = 1, be = 0;
  auto run = [&] { cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &al, w, CUDA_R_16BF, K, x, CUDA_R_16BF, K, &be, o, CUDA_R_16BF, N,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); };
  run(); cudaDeviceSynchronize();
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); std::vector<float> ts;
  int lk = open("/tmp/okc-gpu.lock", O_RDWR | O_CREAT, 0666); flock(lk, LOCK_EX); alarm(20);
  for (int i = 0; i < 60; ++i) { cudaMemsetAsync(fl, i, 200 << 20); cudaEventRecord(a); run(); cudaEventRecord(b); cudaEventSynchronize(b);
    float t; cudaEventElapsedTime(&t, a, b); if (i >= 10) ts.push_back(t * 1000); }
  flock(lk, LOCK_UN); std::sort(ts.begin(), ts.end());
  printf("cuBLAS M=%d: med %.1f us (min %.1f)\n", M, ts[ts.size() / 2], ts[0]);
}
