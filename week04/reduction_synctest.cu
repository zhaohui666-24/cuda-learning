#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
}while(0)

// 用户写法：warp 段也用 __syncthreads（block 级屏障）
__global__ void reduceTwoLoop(const float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x + tid;
    sdata[tid] = (gid<n) ? input[gid] : 0.0f;
    __syncthreads();

    for (int s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    for (int s=32; s>0; s>>=1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid==0) output[blockIdx.x] = sdata[0];
}

int main() {
    int M=256;
    size_t bytes=M*sizeof(float);
    float *hA=(float*)malloc(bytes);
    for (int i=0;i<M;i++) hA[i]=(float)i;   // 递增 0..255

    float *dA,*dOut;
    CUDA_CHECK(cudaMalloc(&dA,bytes));
    CUDA_CHECK(cudaMalloc(&dOut,bytes));
    CUDA_CHECK(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice));

    reduceTwoLoop<<<1,256>>>(dA,dOut,M);
    float h;
    CUDA_CHECK(cudaMemcpy(&h,dOut,sizeof(float),cudaMemcpyDeviceToHost));

    printf("Expected : %d\n", 32640);
    printf("Got      : %.0f  %s\n", h, fabs(h-32640)<0.5?"OK":"WRONG");

    cudaFree(dA); cudaFree(dOut);
    free(hA);
    return 0;
}
