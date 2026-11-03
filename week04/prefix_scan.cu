#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
}while(0)

__global__ void prefixScan(const float* input, float* output, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x + tid;
    s[tid] = (gid<n) ? input[gid] : 0.0f;
    __syncthreads();

    for (int d=1; d<blockDim.x; d<<=1) {
        float v = 0.0f;
        if (tid >= d) v = s[tid - d];   // ① 先读左边 d 处的稳定值
        __syncthreads();                //    等所有线程读完
        s[tid] += v;                    // ② 再写自己的位置
        __syncthreads();                //    等所有线程写完
    }
    if (gid<n) output[gid] = s[tid];
}

int main() {
    int M=256;
    size_t bytes=M*sizeof(float);
    float *hA=(float*)malloc(bytes), *hOut=(float*)malloc(bytes);
    for (int i=0;i<M;i++) hA[i]=(float)i;     // 递增 0..255

    float *dA,*dOut;
    CUDA_CHECK(cudaMalloc(&dA,bytes));
    CUDA_CHECK(cudaMalloc(&dOut,bytes));
    CUDA_CHECK(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice));

    prefixScan<<<1,256>>>(dA,dOut,M);
    CUDA_CHECK(cudaMemcpy(hOut,dOut,bytes,cudaMemcpyDeviceToHost));

    bool ok=true;
    for (int i=0;i<M;i++) {
        float expected = i*(i+1)/2.0f;        // inclusive: 0+1+...+i
        if (fabs(hOut[i]-expected)>1e-3){ok=false;printf("Err at %d: %.0f!=%.0f\n",i,hOut[i],expected);break;}
    }
    printf("Last value: %.0f (should equal reduction sum 32640)\n", hOut[M-1]);
    printf(ok?"PASSED\n":"FAILED\n");

    cudaFree(dA); cudaFree(dOut);
    free(hA); free(hOut);
    return 0;
}
