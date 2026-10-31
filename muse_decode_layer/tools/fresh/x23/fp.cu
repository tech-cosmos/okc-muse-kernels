#include <cstdio>
__global__ void target(float* o){ o[threadIdx.x] = threadIdx.x * 2.f; }
__device__ void* tp = (void*)target;
__global__ void reader(unsigned long long* out){
  const unsigned long long* p = (const unsigned long long*)tp;
  out[0] = (unsigned long long)p;
  for (int i = 0; i < 4; i++) out[1+i] = p[i];
}
int main(){
  unsigned long long* d; cudaMalloc(&d, 64);
  reader<<<1,1>>>(d); cudaError_t e = cudaDeviceSynchronize();
  unsigned long long h[5]={0}; cudaMemcpy(h,d,40,cudaMemcpyDeviceToHost);
  printf("%s addr %llx words %llx %llx %llx %llx\n", cudaGetErrorString(e), h[0],h[1],h[2],h[3],h[4]);
}
