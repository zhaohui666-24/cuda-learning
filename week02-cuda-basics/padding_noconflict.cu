#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
}while(0)

// padding 版：行宽 33，同列错开，无冲突
__global__ void colRead(const float* input, float* output) {
    __shared__ float s[32][33];
    int tid = threadIdx.x;
    s[tid][0] = input[tid];       // 32 线程写同一列
    __syncthreads();
    output[tid] = s[tid][0];
}

int main() {
    int n = 32;                   // 一个 block、一个 warp
    size_t bytes = n*sizeof(float);
    float *hIn=(float*)malloc(bytes), *hOut=(float*)malloc(bytes);
    for (int i=0;i<n;i++) hIn[i]=(float)i;

    float *dIn,*dOut;
    CUDA_CHECK(cudaMalloc(&dIn,bytes));
    CUDA_CHECK(cudaMalloc(&dOut,bytes));
    CUDA_CHECK(cudaMemcpy(dIn,hIn,bytes,cudaMemcpyHostToDevice));

    colRead<<<1,32>>>(dIn,dOut);
    CUDA_CHECK(cudaMemcpy(hOut,dOut,bytes,cudaMemcpyDeviceToHost));

    bool ok=true;
    for (int i=0;i<n;i++) if(hOut[i]!=hIn[i]){ok=false;break;}
    printf("Padded (width=33): %s\n", ok?"PASSED":"FAILED");

    cudaFree(dIn); cudaFree(dOut);
    free(hIn); free(hOut);
    return 0;
}
