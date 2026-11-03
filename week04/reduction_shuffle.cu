#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
}while(0)

__global__ void reduceShuffle(const float* input, float* output, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x + tid;
    int lane   = tid & 31;    // warp 内编号 0..31
    int warpId = tid >> 5;    // 第几个 warp 0..7

    float val = (gid < n) ? input[gid] : 0.0f;

    // ① 每个 warp 内用 shuffle 归约
    for (int offset=16; offset>0; offset>>=1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    // 现在每个 warp 的 lane 0 持有该 warp 的和

    // ② 用一个小 shared 数组收集 8 个 warp 的部分和（只 8 个元素）
    __shared__ float warpSum[8];
    if (lane == 0) warpSum[warpId] = val;
    __syncthreads();

    // ③ 第一个 warp 再用 shuffle 把这 8 个值归约
    if (warpId == 0) {
        val = (lane < 8) ? warpSum[lane] : 0.0f;
        for (int offset=4; offset>0; offset>>=1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane == 0) output[blockIdx.x] = val;
    }
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


    reduceShuffle<<<1,256>>>(dA,dB,M);
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