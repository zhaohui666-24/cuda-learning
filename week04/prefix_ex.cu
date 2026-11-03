#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
}while(0)

__global__ void prefixScanExclusive(const float* input, float* output, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x + tid;

    // —— exclusive 初始化：整体右移一格，第一格补 0 ——
    if (tid == 0) s[0] = 0.0f;
    else          s[tid] = (gid-1 < n) ? input[gid-1] : 0.0f;
    __syncthreads();

    // —— 累加逻辑和 inclusive 完全相同 ——
    for (int d=1; d<blockDim.x; d<<=1) {
        float v = 0.0f;
        if (tid >= d) v = s[tid - d];
        __syncthreads();
        s[tid] += v;
        __syncthreads();
    }

    if (gid < n) output[gid] = s[tid];
}

int main() {
    int M=256;
    size_t bytes=M*sizeof(float);
    float *hA=(float*)malloc(bytes), *hOut=(float*)malloc(bytes);
    for (int i=0;i<M;i++) hA[i]=(float)i;        // 递增 0..255

    float *dA,*dOut;
    CUDA_CHECK(cudaMalloc(&dA,bytes));
    CUDA_CHECK(cudaMalloc(&dOut,bytes));
    CUDA_CHECK(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice));

    prefixScanExclusive<<<1,256>>>(dA,dOut,M);
    CUDA_CHECK(cudaMemcpy(hOut,dOut,bytes,cudaMemcpyDeviceToHost));

    bool ok=true;
    for (int i=0;i<M;i++) {
        // exclusive: y[i] = x[0]+...+x[i-1] = 0+1+...+(i-1) = (i-1)i/2
        float expected = (i==0) ? 0.0f : (i-1)*i/2.0f;
        if (fabs(hOut[i]-expected)>1e-3){ok=false;printf("Err at %d: %.0f!=%.0f\n",i,hOut[i],expected);break;}
    }

    printf("First 6 values: %.0f %.0f %.0f %.0f %.0f %.0f\n",
           hOut[0],hOut[1],hOut[2],hOut[3],hOut[4],hOut[5]);
    printf("Last value: %.0f (sum of x[0..254], excludes x[255])\n", hOut[M-1]);
    printf(ok?"PASSED\n":"FAILED\n");

    cudaFree(dA); cudaFree(dOut);
    free(hA); free(hOut);
    return 0;
}
