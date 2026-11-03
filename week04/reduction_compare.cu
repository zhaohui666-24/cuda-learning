#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
}while(0)

// 版本A：warp 段 for 循环 + 每步 __syncwarp 显式同步
__global__ void reduceForSync(const float* input, float* output, int n) {
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
        __syncwarp();
    }
    if (tid==0) output[blockIdx.x] = sdata[0];
}

// 版本B：经典 volatile 写法，warp 内全线程、手动展开
__global__ void reduceVolatile(const float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x + tid;
    sdata[tid] = (gid<n) ? input[gid] : 0.0f;
    __syncthreads();

    for (int s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid < 32) {
        volatile float* v = sdata;
        v[tid] += v[tid+32];
        v[tid] += v[tid+16];
        v[tid] += v[tid+8];
        v[tid] += v[tid+4];
        v[tid] += v[tid+2];
        v[tid] += v[tid+1];
    }
    if (tid==0) output[blockIdx.x] = sdata[0];
}

int main() {
    int M=256;
    size_t bytes=M*sizeof(float);
    float *hA=(float*)malloc(bytes);
    for (int i=0;i<M;i++) hA[i]=(float)i;

    float *dA,*dOut;
    CUDA_CHECK(cudaMalloc(&dA,bytes));
    CUDA_CHECK(cudaMalloc(&dOut,bytes));
    CUDA_CHECK(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice));

    float h1,h2;
    reduceForSync<<<1,256>>>(dA,dOut,M);
    CUDA_CHECK(cudaMemcpy(&h1,dOut,sizeof(float),cudaMemcpyDeviceToHost));

    reduceVolatile<<<1,256>>>(dA,dOut,M);
    CUDA_CHECK(cudaMemcpy(&h2,dOut,sizeof(float),cudaMemcpyDeviceToHost));

    printf("Expected        : %d\n", 32640);
    printf("For + syncwarp  : %.0f  %s\n", h1, fabs(h1-32640)<0.5?"OK":"WRONG");
    printf("Volatile classic: %.0f  %s\n", h2, fabs(h2-32640)<0.5?"OK":"WRONG");

    cudaFree(dA); cudaFree(dOut);
    free(hA);
    return 0;
}
