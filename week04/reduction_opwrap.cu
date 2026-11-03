#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
}while(0)

__global__ void reduceNaive(const float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x + tid;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;  // ① 每线程搬一个元素进 shared
    __syncthreads();

    for (int s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s)  sdata[tid] += sdata[tid+s];
        __syncthreads();
    }

    for (int s=32; s>0; s>>=1) {
        if (tid < s)  sdata[tid] += sdata[tid+s];
        __syncwarp();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];  // ③ 每 block 一个部分和
}

int main() {
    int M=256;
    size_t sA=M*4,sB=M*4;

    float *hA=(float*)malloc(sA),*hB=(float*)malloc(sB);
    for (int i=0;i<M;i++) hA[i]=1.0f;

    float *dA,*dB;
    CUDA_CHECK(cudaMalloc(&dA,sA));
    CUDA_CHECK(cudaMalloc(&dB,sB));
    CUDA_CHECK(cudaMemcpy(dA,hA,sA,cudaMemcpyHostToDevice));


    reduceNaive<<<1,256>>>(dA,dB,M);
    CUDA_CHECK(cudaGetLastError());



    CUDA_CHECK(cudaMemcpy(hB,dB,sB,cudaMemcpyDeviceToHost));

    float expected = 0.0f;              // 普通 float 变量，不是指针
    for (int i=0;i<M;i++) expected += hA[i];

    if (fabs(hB[0]-expected) < 1e-5) printf("PASSED\n");
    else printf("FAILED: %f != %f\n", hB[0], expected);
    cudaFree(dA); cudaFree(dB);
    free(hA); free(hB);
    return 0;
}
